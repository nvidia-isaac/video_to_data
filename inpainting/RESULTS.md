# Visual-inpainting investigation results

Date: 2026-07-22

## Controlled comparison

The first experiment varies only the hand-trajectory source: TACO motion-
capture ground truth, Video2Data WiLoR, or Phantom Grounding DINO + HaMeR.
Every condition consumes the same visually reviewed SAM2 two-arm mask, pinned
E2FGVI-HQ implementation, 960-pixel processing cap, official TACO camera,
Vega + Sharpa renderer, and TACO object-depth compositor. The learned hand
predictions and Sharpa retargeting never consume GT hand tracks. The current
experiment is nevertheless not pure-RGB end to end: every condition uses the
official camera metadata, and both learned final composites deliberately reuse
GT tool/target meshes and per-frame 6-DoF poses to render one shared occluder
depth pass. This isolates hand-tracker differences, but makes object occlusion
an oracle component of the controlled comparison.

All per-condition stage videos retain the 1920x1080, 30 FPS source geometry
and exact source frame count; labelled comparison grids deliberately tile
those streams at review resolution. The definitive common-clip comparison
(`253`) fingerprints model
checkpoints, licensed MANO files, camera inputs, public weights, immutable
container images, implementation sources, robot assets, and committed outputs.
Inference containers run offline after explicit acquisition. The longer `105`
result now provides a second five-way comparison using current robot-render,
composite, and grid provenance. Its learned tracking and Sharpa inputs retain
legacy sidecars, so it remains supporting evidence rather than being mislabeled
as current v2 tracking provenance. The incomplete `060` learned tracks are
retained for failure analysis.

The definitive `253` E2FGVI output was rerun with immutable image
`sha256:398b54800eebd0343ec27ba86c1a59829cb7439ce9418ec533744e837558ebbc`;
its SHA-256 remained byte-for-byte identical
(`250a7e7a5a1de96995b2d32d68a56b4dfac1413a492415237e695982b1ad6c08`).
The existing `060` and `105` outputs predate immutable-image recording. Their
sidecars now record output bytes and hashes but explicitly label container
identity `legacy_unrecorded`; the migration does not invent provenance. Object
mask/depth bundles for all three clips were rerun with immutable renderer image
`sha256:86ca30a0310c25fa3c0eb5e28a282b21e8a95b151e7011a00bbe4f3bbc06ed63`.
All six arrays were byte-identical to the reviewed generation, after which the
GT composites/grids and both definitive `253` learned composites/grids were
republished against the refreshed metadata.

## Completion summary

| Clip | Frames | Ground truth | Video2Data | Phantom |
|---|---:|---|---|---|
| `060` kettle/plate | 155 | Full result | Tracking `155/146`; render intentionally blocked by 9 right-hand gaps | Tracking `155/148`; render intentionally blocked by 7 right-hand gaps |
| `105` knife/plate | 152 | Full result | Full current render/composite, `152/152` hands (legacy tracking/Sharpa sidecars) | Full current render/composite, `152/152` hands (legacy tracking/Sharpa sidecars) |
| `253` brush/cup | 74 | Full result | Definitive v2 full result, `74/74` hands | Definitive v2 full result, `74/74` hands |

Counts are left/right source-valid frames. Learned gaps remain NaN and are not
held, interpolated, or filled from solver state. The strict renderer therefore
rejects 060 rather than hallucinating missing robot poses.

## Tracking accuracy on the common 253 clip

The deterministic evaluator converts each track independently through the
official camera, intersects validity masks, and compares MANO-order joints.
Values below are means over 74/74 valid frames per side; the two report JSONs
fingerprint the exact prediction, ground truth, video, and camera inputs.

| Tracker | Side | Wrist 3D | 21-joint MPJPE | Projected 2D MPJPE | Mean-joint temporal step | GT temporal step |
|---|---|---:|---:|---:|---:|---:|
| Video2Data | Left | 66.394 mm | 57.458 mm | 37.025 px | 8.271 mm | 4.354 mm |
| Video2Data | Right | 65.550 mm | 54.557 mm | 33.242 px | 13.109 mm | 7.371 mm |
| Phantom | Left | 73.952 mm | 63.451 mm | 35.854 px | 8.233 mm | 4.354 mm |
| Phantom | Right | 88.429 mm | 76.784 mm | 35.308 px | 13.631 mm | 7.371 mm |

Video2Data is better in metric 3D on both hands and in right-hand 2D. Phantom
is slightly better in left-hand 2D and slightly smoother on the left;
Video2Data is slightly smoother on the right. Both learned methods are visibly
noisier than motion capture, especially on the right hand. Their broadly
similar 2D error but different 3D error also shows why image overlays alone are
insufficient for choosing a robot trajectory.

## Tracking accuracy on the 105 clip

The same evaluator covers all 152 paired-valid frames per hand on `105`.

| Tracker | Side | Wrist 3D | 21-joint MPJPE | Projected 2D MPJPE | Mean-joint temporal step | GT temporal step |
|---|---|---:|---:|---:|---:|---:|
| Video2Data | Left | 122.906 mm | 112.714 mm | 44.304 px | 8.535 mm | 3.410 mm |
| Video2Data | Right | 90.738 mm | 78.192 mm | 35.540 px | 11.701 mm | 7.195 mm |
| Phantom | Left | 130.456 mm | 117.787 mm | 43.337 px | 10.622 mm | 3.410 mm |
| Phantom | Right | 104.195 mm | 91.000 mm | 33.673 px | 11.776 mm | 7.195 mm |

Video2Data is better in calibrated metric 3D on both hands. Phantom is slightly
better after projection into the image plane on both hands. The 2D metric is a
projection of each method's 3D MANO joints through the common camera, not a
score on an independently selected 2D detection.

## End-to-end robot result on 253

| Tracker | Vega max residual | Vega p95 residual | Max arm step | Robot pixels hidden behind objects |
|---|---:|---:|---:|---:|
| Ground truth | 0.063 mm | 0.004 mm | 0.225 rad/frame | 271,683 |
| Video2Data | 10.935 mm | 0.281 mm | 0.307 rad/frame | 557,650 |
| Phantom | 24.706 mm | 0.277 mm | 0.346 rad/frame | 548,627 |

The learned p95 residuals remain below 0.3 mm, but each has an isolated larger
IK attachment outlier. Video2Data passes an explicit 12 mm ceiling. Phantom
first failed that same ceiling, then completed under a separately declared
30 mm condition after review; its 24.706 mm maximum is retained as a result,
not averaged away. The Sharpa frame-task gate is likewise an explicit 70 mm
catastrophic-solution guard rather than a claim of fingertip convergence, and
every source-valid frame passes it.

Depth-aware compositing is material rather than cosmetic. Hundreds of
thousands of robot pixels lie behind the real tool or target in every 253
condition. A hard overlay would put those pixels incorrectly in front; the
final results use metric object depth with a 3 mm guard.

## End-to-end robot result on 105

| Tracker | Vega max residual | Vega p95 residual | Max arm step | Robot pixels hidden behind objects |
|---|---:|---:|---:|---:|
| Ground truth | 0.251 mm | 0.088 mm | 0.156 rad/frame | 147,752 |
| Video2Data | 0.198 mm | 0.008 mm | 0.389 rad/frame | 1,628,800 |
| Phantom | 0.043 mm | 0.006 mm | 0.276 rad/frame | 1,805,741 |

Both learned conditions pass their declared gates for all 152 frames. Strict
resume then validates all three render/composite/grid stages as
`skipped_complete` for each condition.

## Visual findings

- The calibrated learned skeletons follow both visible hands at the start,
  middle, and end of all three clips. Phantom's corrected native-side MANO
  axis conversion matches manotorch `AxisLayerFK` to `4.9e-7` maximum matrix
  error before Sharpa retargeting.
- Both 253 learned robot renders place the Sharpa hands near the human contact
  regions throughout the clip. Video2Data is somewhat closer in metric wrist
  space; Phantom's worst downstream IK outlier is the clearest failure signal.
- Both 105 learned results remain stable for all 152 frames. Video2Data stays
  below 0.390 rad/frame and Phantom below 0.276 rad/frame, providing a second
  full three-condition comparison rather than relying only on the short 253
  clip.
- E2FGVI removes both arms while retaining the kettle, plate, knife, brush, and
  cup. It leaves some original cast shadows and mild table/floor texture
  smearing; those are shared baseline limitations, not tracker differences.
- The current robot renderer prioritizes calibrated geometry and articulation
  over photorealistic lighting, material matching, or synthesized shadows.

## Review artifacts

- Final five-way `253` comparison (source, shared E2FGVI, GT, Video2Data,
  Phantom):
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_253_tracker_comparison.mp4`
- Start/middle/end contact sheet for the five-way comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_253_tracker_comparison_contact.jpg`
- Final five-way `105` comparison (source, shared E2FGVI, GT, Video2Data,
  Phantom):
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_105_tracker_comparison.mp4`
- Start/middle/end contact sheet for the `105` comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_105_tracker_comparison_contact.jpg`
- GT synchronized result across all three clips:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_gt_depth_overlay_all_sequences.mp4`
- Video2Data 253 full comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/v2d/final_comparison_grid.mp4`
- Phantom 253 full comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/phantom/final_comparison_grid.mp4`
- Video2Data 105 full comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/v2d/final_comparison_grid.mp4`
- Phantom 105 full comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/phantom/final_comparison_grid.mp4`
- Learned tracking overlays:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/{v2d/audit/tracking_overlay.mp4,phantom/tracking/hand_overlay.mp4}`
- Deterministic `105` and `253` evaluation reports:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/{v2d,phantom}/tracking/evaluation_vs_ground_truth.json`

The next useful ablations are RGB-estimated object depth, tracker-specific arm
masks, no-inpainting versus E2FGVI under learned trajectories, and Cosmos3.
For depth, the staged comparison should first use the GT object mask with
estimated metric depth, then an RGB-derived object mask and depth, and finally
an RGB-reconstructed mesh/pose depth pass. These should remain separate
controlled changes rather than being folded into this tracking comparison.
