# GasVid Rights and Metadata Inquiry

Status: `draft_not_sent`

Prepared: 2026-07-31

Primary recipient: Jingfan Wang (`jingfan@stanford.edu`), corresponding author
of the original GasVid paper.

Copy: Adam Brandt (`abrandt@stanford.edu`), current Stanford Environmental
Assessment and Optimization Group contact.

Subject: Permission and metadata clarification for academic GasVid analysis

## Message

Dear Dr. Wang and Professor Brandt,

I am working on a non-commercial research demonstrator for mechanically
constrained active sensing of methane plumes. We would like to use the public
GasVid videos linked by the LangGas project as externally recorded input. The
planner would receive only causally sampled grayscale pixels; the complete
videos would remain local and would not be committed to our software
repository.

Could you please clarify whether you permit the following uses of GasVid?

1. Local academic download, decoding, and analysis of the videos.
2. Extraction of bounded frame intervals for experiments.
3. Creation and local storage of derived background/noise calibration maps,
   structure-suppressed grayscale evidence, and diagnostic plume masks.
4. Publication of aggregate metrics and a small number of illustrative
   frames, figures, or short rendered comparison clips.
5. Redistribution, if any, of raw videos, extracted frames, derived images,
   masks, or calibration products. We are happy to keep all media private if
   redistribution is not permitted.

We would also appreciate clarification of the experimental metadata:

- Which video filenames belong to separator pad 1 and separator pad 2?
- Does each `Leak 0`/class-0 interval represent a true no-release background,
  despite Table 1 listing a small nonzero class-0 flow rate?
- Are the eight classes always three minutes long, including the two logging
  anomalies in `MOV_2568` and `MOV_2582`?
- Are synchronized wind measurements, source pixel/corridor annotations, or a
  more complete per-video metadata table available?
- Is there a preferred citation or formal dataset licence/terms document?

We will preserve the paper-defined acquisition split, keep evaluation data out
of planner development, and describe GasVid as compressed OGI imagery rather
than calibrated gas concentration or clean-sensor data.

Thank you for any guidance you can provide.

Best regards,

[Name and affiliation]

## Evidence handling

- Do not change `rights_status` from `unverified` based on silence or public
  downloadability.
- Save any response or formal terms outside the repository if it contains
  personal correspondence; record only its date, scope, stable reference, and
  hash where appropriate.
- Preparation may proceed only after the response explicitly permits local
  processing and creation of the required derivatives.
- Raw redistribution and publication of derived imagery are independent
  permissions and must be recorded separately in the rights provenance.
