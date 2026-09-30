**Visual evidence companion — May versus July**

These are sampled frames from existing rendered artifacts, not newly generated experimental results. All nine videos listed in the [decode manifest](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-manifest.json>) decoded completely. Five frames per video were selected across its playback duration for visual inspection. Frame timestamps below are playback time unless the rendered label says otherwise. See the [full review](<C:/Users/05mus/Documents/New project 2/docs/reviews/2026-09-12-scope-review.md>) for code and report checks.

**May: a visible scan unfolding**

At frame 189, 15.75 seconds into the [May adapted-sweep video recovered from commit 1fcca43](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/may04/methane-tilted-adapted-live-plume-realtime-scan.mp4>), only part of the selected sweep has been filled. This is a useful presentation of single-detector scanning. The caption says the policy was selected at update 20; the reveal itself does not establish that the policy updates after each measurement.

![May scan accumulating at 15.75 seconds](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/methane-tilted-adapted-live-plume-realtime-scan-0189.jpg>)

[Five-frame sequence](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/methane-tilted-adapted-live-plume-realtime-scan-contact.jpg>).

**May: visible focusing, with a hidden timing problem**

The first frame of the [May full-scene proof](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/may04/methane-full-scene-start-proof.mp4>) labels the completed scan as 336 seconds. The whole 46-frame movie lasts only 3.83 seconds. Time compression can be legitimate, but this particular historical loop also starts successive physical scans only a video frame apart and scores against the next frame. The code, not playback compression alone, establishes that defect.

![May full-scene initial scan](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/methane-full-scene-start-proof-0000.jpg>)

[Five-frame sequence](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/methane-full-scene-start-proof-contact.jpg>).

**July: the mechanical scan and colored belief tell different stories**

Frame 22 of the [source-aware demo](<C:/Users/05mus/Documents/New project 2/outputs/source-aware-46-cycle-causal-methane.mp4>) shows a cyan scan mask enclosing the plume, while colored density peaks occupy smaller regions. The accompanying report’s 100% actual-sweep recall concerns the cyan sweep, not calibration of the colored field. This run uses a 5 ms full-height pass and zero horizontal step delay, so visual responsiveness is not hardware evidence.

![July source-aware scan and belief](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/source-aware-46-cycle-causal-methane-0022.jpg>)

[Five-frame sequence](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/source-aware-46-cycle-causal-methane-contact.jpg>).

**July OGI: measurements are hidden by the renderer**

Frame 3 of the [raw propane OGI diagnostic](<C:/Users/05mus/Documents/New project 2/data/recorded/outputs/usu-r0c-v1-rendered.mp4>) shows almost only yellow scan stripes and a broad red wash. There are eight frames total, 0.535 seconds of playback, representing 36.22 seconds of modeled scan execution.

![July raw OGI diagnostic](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/usu-r0c-v1-rendered-0003.jpg>)

The scan stripes are not a trustworthy view of measurement variation: a one-pixel outline repaints each one-pixel sampled column. In the [controlled rendering probe](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/render-probes.json>), all 192 measurements changed from 0 to 255 but zero displayed pixels changed. That defect also exists in May’s renderer.

The red field is max-normalized relative density, which can saturate for a uniform uncertain distribution. No spatial truth exists in this recorded run, despite the generic legend’s “true plume envelope” wording. Its report contains zero evaluated predictions, not evidence of measured zero plume recall.

[Five-frame sequence](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/usu-r0c-v1-rendered-contact.jpg>).

**July 31: the corrected fast comparison still exposes allocation problems**

Frame 44 of the [paired development diagnostic](<C:/Users/05mus/Documents/New project 2/outputs/fair-continuous-stream-preview.mp4>) shows detached adaptive scan columns, including a far-right strip, while the static corridor uses a compact region. The full plume is displayed as evaluator-only context, not as the scanner’s observations.

![Detached adaptive columns in the July paired diagnostic](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/fair-continuous-stream-preview-0044.jpg>)

The saved report gives mean completion recall of 80.92% adaptive versus 95.12% static, at the same modeled scan duration of 0.1091 seconds. Neither policy triggered reacquisition in this case. This supports a remaining allocation problem; it is one development case and cannot settle general policy performance.

[Five-frame sequence](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-frames/fair-continuous-stream-preview-contact.jpg>).

The next useful visualization should accumulate actual scalar measurements on a physical clock, show the local prior/likelihood/posterior change when data arrives, then show the predicted next sweep. Rendering should make the sensing model inspectable rather than substitute an appealing heatmap for that model.
