"""Static scientific inspection of matched-evidence forecasts (not a renderer redesign)."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    root=args.directory
    data=json.loads((root/"diagnosis/fixed-evidence.json").read_text())
    for profile in ("coarse-full-startup","double-prior-startup"):
        rows=[r for r in data["rows"] if r["source_run"].startswith(profile) and r["decision_index"]==1]
        fig, axes=plt.subplots(2,3,figsize=(11,6.5),layout="constrained")
        for index,row in enumerate(rows):
            forecast=next(f for f in row["forecasts"] if f["offset_seconds"]==6)
            for column,(field,label,upper) in enumerate((("mean","Predicted signal",1.),("std","Signal standard deviation",.2),("excess_density","Expected excess above 0.15",.5))):
                values=np.asarray(forecast[field])
                im=axes[index,column].imshow(values,extent=(0,1,1,0),vmin=0,vmax=upper,cmap="viridis",aspect="auto")
                axes[index,column].set(title=label,xlabel="Horizontal FOV",ylabel=f"{row['estimator']} / vertical FOV")
                fig.colorbar(im,ax=axes[index,column],shrink=.7)
            if row["motion"]:
                bias=row["motion"]["inferred_bias_yx"]
                axes[index,0].text(.02,.97,f"Bias y,x ({bias[0]:+.4f}, {bias[1]:+.4f}) FOV/s\nMax hypothesis weight {row['motion']['maximum_weight']:.1%}",
                    transform=axes[index,0].transAxes,fontsize=8,va="top",bbox={"facecolor":"white","alpha":.9,"edgecolor":"none"})
        fig.suptitle(f"{profile}: identical delivered evidence at {rows[0]['at_seconds']:.3f} s; forecast +6 s\n"
                     f"Initial {data['quadrature_order']**2}-hypothesis diagnostic; synthetic units; no superiority claim",fontsize=12)
        for suffix in ("png","pdf"):
            fig.savefig(root/"diagnosis"/f"{profile}-fixed-forecast.{suffix}",dpi=150)
        plt.close(fig)
    fig, axes=plt.subplots(2,3,figsize=(11,7),layout="constrained")
    for row,profile in enumerate(("coarse-full-startup","double-prior-startup")):
        for column,order in enumerate((3,5,7)):
            path=root/"convergence"/f"order-{order}"/f"{profile}-wind-error-threshold90_segment_v2-seed-0-motion"/"report.json"
            if not path.exists():
                axes[row,column].set_axis_off()
                axes[row,column].set_title(f"{order**2} hypotheses: not run")
                continue
            report=json.loads(path.read_text())
            posterior=report["scan_schedule"][-1]["posterior_motion"]
            offsets=np.asarray(posterior["bias_offsets_yx"])
            weights=np.asarray(posterior["weights"])
            source=report["configuration"]["source"]
            correction=np.asarray(source["wind_velocity"])*source["fps"]/(np.asarray(source["shape"])-1)-report["configuration"]["wind"]["velocity_yx"]
            ax=axes[row,column]
            ax.scatter(offsets[:,1],offsets[:,0],s=15,color="lightgray",label="All retained hypotheses")
            ax.scatter(offsets[:,1],offsets[:,0],s=600*weights,color="#2855a5",alpha=.85,label="Area = posterior weight")
            ax.scatter(correction[1],correction[0],s=90,marker="x",color="#c95010",linewidths=2,label="Generator wind correction")
            ax.set(xlim=(-.045,.045),ylim=(-.045,.045),title=f"{order**2} hypotheses / max weight {weights.max():.1%}",
                   xlabel="Horizontal bias (FOV/s)",ylabel=f"{'Coarse' if row==0 else 'Doubled'} / vertical bias (FOV/s)")
            ax.grid(alpha=.2); ax.set_aspect("equal")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=9)
    fig.suptitle("Final wind posteriors in the refinement runs\nDiscrete bias approximations; each run executes its own path",fontsize=12)
    for suffix in ("png","pdf"):
        fig.savefig(root/"diagnosis"/f"refinement-wind-posteriors.{suffix}",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    main()
