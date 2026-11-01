"""Show how two fixed source-location assumptions change the same-data map.

Presentation-only observer: no policy change, no extra measurements, and no
renderer-driven filter steps. Ground truth is used only in the left panel.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np

from plume_focus.full_fov_execution import decode_config, run_full_fov, verify_replay
from plume_focus.sensing_contracts import CausalWind, SensorReading
from plume_focus.signal_belief import grid_positions, interpolation_stencil
from plume_focus.source_prior_belief import SourcePriorBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, help='Optional existing recorded demo')
    parser.add_argument('--config', type=Path,
                        default=Path('configs/evidence-gated-focus/coarse-demo.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/source-assumptions-demo'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.record:
        record = json.loads(args.record.read_text())
        shutil.copyfile(args.record, args.output / 'record.json')
    else:
        record = run_full_fov(decode_config(json.loads(args.config.read_text())))
        (args.output / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    verify_replay(record)
    config = decode_config(record['configuration'])
    correct = np.array(config.session.source_position)
    offset = np.array([-.10, .18])
    incorrect = correct + offset
    if np.any((incorrect < 0) | (incorrect > 1)):
        raise ValueError('This demo offset requires both locations inside the field of view')
    bank = SourcePriorBank(config.model, np.array([correct, incorrect]), np.eye(2),
                           half_life_seconds=config.session.half_life_seconds,
                           transport_id=config.session.transport_id)
    shape = (20, 32)
    points = grid_positions(shape)
    ids, weights = interpolation_stencil(points, config.model.grid_shape)
    source = config.build_source()  # Observer only: never supplied to inference.
    wind = CausalWind()
    times = np.arange(0, config.duration_seconds, .5)
    times = np.append(times, config.duration_seconds)
    fields, truths, counts, readings = [], [], [], []
    events = iter(record['events'])
    event = next(events, None)
    for time in times:
        while event is not None and event['at_seconds'] <= time:
            at = event['at_seconds']
            if event['operation'] == 'plan_next':
                bank.advance(at, wind)
                wind = CausalWind.from_dict(event['wind'])
            elif event['operation'] == 'ingest_readings':
                if len(event['readings']) != 1:
                    raise ValueError('This renderer expects the original singleton-reading demo')
                reading = SensorReading.from_dict(event['readings'][0])
                bank.assimilate(reading, wind)
                readings.append([reading.available_at_seconds, reading.value])
            elif event['operation'] == 'finish_scan':
                bank.advance(at, wind)
            else:
                raise ValueError('Unrecognized event')
            event = next(events, None)
        # A pure forecast for display; never changes filtering or future updates.
        means = bank.forecast_aggregate(float(time), wind)
        fields.append(sum(means[:, ids[:, k]] * weights[:, k] for k in range(4)))
        truths.append(source.signal_at(np.full(len(points), time), points))
        counts.append(len(readings))
    fields, truths, readings = np.asarray(fields), np.asarray(truths), np.asarray(readings)
    np.savez_compressed(args.output / 'frames.npz', time=times, field=fields,
                        truth=truths, readings=readings, delivered_count=counts)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    fig.suptitle('Same readings. Different source assumptions.', y=.97, fontsize=23)
    status = fig.text(.5, .91, '', ha='center', fontsize=13)
    axes = [fig.add_axes([.055+i*.305, .34, .26, .48]) for i in range(3)]
    vmin = min(0., float(fields.min()), float(truths.min()))
    vmax = max(float(fields.max()), float(truths.max()))
    titles = ['What is actually simulated', 'Map with correct source hint',
              'Map with incorrect source hint']
    images = []
    for ax, title in zip(axes, titles):
        images.append(ax.imshow(np.zeros(shape), origin='lower', extent=(0, 1, 0, 1),
                               cmap='magma', vmin=vmin, vmax=vmax, interpolation='nearest'))
        ax.set(title=title, xlabel='Horizontal position / field of view')
        ax.set_aspect('equal')
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    axes[0].set_ylabel('Vertical position / field of view')
    axes[0].plot(correct[1], correct[0], 'o', color='#00deef', ms=8, mfc='none', mew=2)
    axes[1].plot(correct[1], correct[0], '+', color='#00deef', ms=14, mew=2)
    axes[2].plot(incorrect[1], incorrect[0], '+', color='#00deef', ms=14, mew=2)
    colorbar = fig.colorbar(images[0], cax=fig.add_axes([.944, .40, .010, .30]))
    colorbar.ax.set_title('Signal', fontsize=10, pad=8)
    colorbar.ax.tick_params(labelsize=9)
    fig.text(.5, .855, 'Circle: actual source (evaluation only).  Plus: fixed source hint given to the estimator.',
             ha='center', fontsize=11)
    signal_ax = fig.add_axes([.075, .15, .85, .105])
    signal_ax.set(xlim=(0, config.duration_seconds), ylim=(min(0., readings[:, 1].min()),
                    max(.01, readings[:, 1].max())*1.1),
                  xlabel='Simulation time (s)', ylabel='Sensor value')
    signal_ax.spines[['top', 'right']].set_visible(False)
    line, = signal_ax.plot([], [], '.', ms=3, color='#526171')
    signal_ax.set_title('Identical delivered readings supplied to both estimators', fontsize=12)
    fig.text(.5, .058, 'Synthetic fixed-path comparison. Maps are estimates, not camera images. '
             'Source hints stay fixed; uncertainty is not displayed.', ha='center', fontsize=10)
    fig.text(.5, .025, 'Playback: 2x simulation time. No new scan decisions are made in this comparison.',
             ha='center', fontsize=10)

    def update(i):
        for im, a in zip(images, [truths[i], fields[i, 0], fields[i, 1]]):
            im.set_data(a.reshape(shape))
        n = counts[i]
        line.set_data(readings[:n, 0], readings[:n, 1])
        status.set_text(f'Simulation time: {times[i]:.1f} s   |   Readings received: {n}')

    update(len(times)-1)
    fig.savefig(args.output / 'same-readings-final.png', dpi=140)
    gif = PillowWriter(fps=4)
    with gif.saving(fig, str(args.output / 'same-readings-demo.gif'), dpi=90):
        for i in range(len(times)):
            update(i); gif.grab_frame()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ffmpeg:
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
        video = FFMpegWriter(fps=4, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        with video.saving(fig, str(args.output / 'same-readings-demo.mp4'), dpi=100):
            for i in range(len(times)):
                update(i); video.grab_frame()
    plt.close(fig)
    manifest = {
        'question': 'Can a wrong source hint change the map inferred from the same readings?',
        'configuration': record['configuration'], 'configuration_hash': record['configuration_hash'],
        'correct_source_yx': correct.tolist(), 'incorrect_source_yx': incorrect.tolist(),
        'frame_shape': list(shape), 'display_sample_seconds': .5, 'playback_fps': 4,
        'playback_speed': 2, 'physical_horizon_seconds': config.duration_seconds,
        'reading_count': len(readings), 'public_replay': 'passed',
        'scope': 'Original fixed source-anchored path; two delta source assumptions. '
                 'Rendering-only comparison, no new policy or physical concentration claim.',
        'record_sha256': hashlib.sha256((args.output/'record.json').read_bytes()).hexdigest(),
        'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in args.output.iterdir() if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'readings': len(readings), 'frames': len(times),
                      'mp4': bool(ffmpeg), 'public_replay': 'passed'}))


if __name__ == '__main__':
    main()
