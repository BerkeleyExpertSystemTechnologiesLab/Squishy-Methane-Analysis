"""Observer-only movie of the first preregistered high-input closed-loop pair."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imageio_ffmpeg
from plume_focus.full_fov_execution import decode_plan
from plume_focus.sensing_simulation import pose_at


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path)
    p.add_argument('--closed',required=True,type=Path);args=p.parse_args()
    cfg=json.loads((args.closed/'config.json').read_text());summary=json.loads((args.closed/'summary.json').read_text())
    # Select from the declared input condition, never from performance outcomes.
    case_index=next(i for i,r in enumerate(summary) if r['case']['amount']==1.5)
    case=summary[case_index]['case'];shape=tuple(cfg['closed_loop']['grid_shape'])
    env=args.study/f'grid-{shape[0]}x{shape[1]}'/'environment'
    info=json.loads((env/'environment.json').read_text())
    det=np.load(env/'deterministic.npy',mmap_mode='r');noise=np.load(env/'noise.npy',mmap_mode='r')
    si=next(i for i,s in enumerate(info['scenes']) if s['name']==case['scene']);ni=info['seeds'].index(case['seed'])
    arrays=[np.load(args.closed/f'case-{case_index:02d}-{method}'/'arrays.npz') for method in ['fixed','joint']]
    events=[json.loads((args.closed/f'case-{case_index:02d}-{method}'/'events.json').read_text()) for method in ['fixed','joint']]
    out=args.closed/'demo-first-high-input';out.mkdir(exist_ok=False)
    times=np.append(arrays[0]['times'],cfg['closed_duration_seconds'])
    assert np.array_equal(arrays[0]['times'],arrays[1]['times'])
    fields=[np.concatenate([a['frames'],a['final_mean'][None]]) for a in arrays]
    truth=np.stack([det[int(round(t/info['tick'])),si]*case['amount']+noise[int(round(t/info['tick'])),ni] for t in times])
    vmax=max(float(x.max()) for x in [truth,*fields]);vmin=min(float(x.min()) for x in [truth,*fields])
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained');images=[]
    for ax,title in zip(axes,['Simulated truth','Fixed default source amount','Joint location + amount']):
        images.append(ax.imshow(np.zeros(shape),origin='lower',extent=[0,1,0,1],cmap='viridis',vmin=vmin,vmax=vmax,aspect='auto',interpolation='nearest'))
        ax.set(title=title,xlabel='Normalized x',ylabel='Normalized y',xlim=(0,1),ylim=(0,1))
    fig.colorbar(images[0],ax=axes,shrink=.8,label='Synthetic field signal (one shared scale)')
    points=[ax.plot([],[],'o',color='#ff884d',ms=3,alpha=.75)[0] for ax in axes[1:]]
    poses=[ax.plot([],[],'wx',ms=10,mew=2)[0] for ax in axes[1:]]
    plans=[[decode_plan(e['plan']) for e in es if e['op']=='plan'] for es in events]
    readings=[[e['reading'] for e in es if e['op']=='reading'] for es in events]
    header=fig.suptitle('',fontsize=12)
    fig.canvas.draw();width,height=fig.canvas.get_width_height()
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','4','-i','-','-an','-vcodec','libx264','-pix_fmt','yuv420p','-crf','23','-movflags','+faststart',str(out/'loop.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);gif=[]
    for fi,t in enumerate(times):
        statuses=[]
        for image,field in zip(images,[truth[fi],fields[0][fi],fields[1][fi]]):image.set_data(field.reshape(shape))
        for mi in range(2):
            available=[r for r in readings[mi] if t-3<=r['available_at_seconds']<=t]
            points[mi].set_data([r['position_yx'][1] for r in available],[r['position_yx'][0] for r in available])
            active=next((p for p in plans[mi] if p.samples and p.started_at_seconds<=t<=p.completed_at_seconds),None)
            if active:
                y,x=pose_at(active,t);poses[mi].set_data([x],[y]);statuses.append(active.mode)
            else:
                poses[mi].set_data([],[]);statuses.append('waiting / no legal scan fits budget')
        header.set_text(f"Physical time {t:5.1f} / 60 s | {case['scene']}, true source input 1.5x\nFixed: {statuses[0]} | Joint: {statuses[1]} | Playback: 2x physical speed")
        fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
        proc.stdin.write(rgb.tobytes())
        gif.append(Image.fromarray(rgb).quantize(colors=64))
        if fi in [0,len(times)-1]:Image.fromarray(rgb).save(out/f'frame-{fi:03d}.png')
    proc.stdin.close();stderr=proc.stderr.read();code=proc.wait()
    if code:raise RuntimeError(stderr.decode())
    gif[0].save(out/'loop.gif',save_all=True,append_images=gif[1:],duration=250,loop=0,optimize=False)
    plt.close(fig)
    metadata={'case_index':case_index,'case':case,'selection':'first declared high-input pair, independent of results',
        'physical_seconds':60,'frames':len(times),'playback_fps':4,'playback_seconds':len(times)/4,
        'orange_points':'Readings delivered during the preceding 3 physical seconds',
        'white_cross':'Current mechanical pointing position; absent while idle',
        'truth_render_only':True,'original_policy_unchanged':True}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(metadata))


if __name__=='__main__':main()
