# Visual-inpainting investigation results

Date: 2026-07-23

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

## Parallel-jaw embodiment extension

The completed extension keeps the same source RGB, arm mask, E2FGVI output,
official camera, GT object-depth occluder, and GT/Video2Data/Phantom hand
tracks, but replaces Vega + Sharpa with two bimanual parallel-jaw robots:
Galbot Golf and YAM. A shared robot-neutral target derives aperture from the
thumb/index tip distance, translation from the thumb/middle midpoint, and
orientation from the same thumb/index/palm geometry used by Phantom. Phantom's
translation/aperture GP smoothing, Gaussian-SLERP orientation smoothing, and
20% sequence-relative grasp-span cap are applied identically to all three
trackers before embodiment-specific aperture mapping.

The source assets are pinned to Galbot commit
`b311f5ca1acf506e9b7026397e2c74fb2db11df6` and YAMLab commit
`ec0455d2b4ce35f21fc126418ea5e74ac567133d`; the reviewed RoboLab examples are
MR 62 commit `8224d5fb8a2a3d21ce445bb198476c1faa4d69e6` and MR 68 commit
`543a08bf3b46aa8fb2abc79ffba09cf4d09e09ae`. Derived render/IK bundles preserve
the physical Galbot `[0, 124.909]` mm and YAM `[2.004, 94.901]` mm aperture
ranges. No commanded aperture clips in any of the 12 production renders.

Every tracker reuses the same GT-derived Vega hub transform within a clip.
This avoids improving a condition by moving the robot body, but exposes a real
Galbot-105 limitation: no tested common mount satisfies the 20-degree
orientation gate. Those three runs use a disclosed 55-degree / 0.55 rad policy
after stress-frame review; Galbot-253 and all YAM runs retain the strict
10 mm / 20-degree / 0.4 rad policy.

| Robot | Clip | Tracker | Max position residual | Max orientation residual | Max arm step | Gate result |
|---|---|---|---:|---:|---:|---|
| Galbot Golf | `105` | Ground truth | 0.565 mm | 34.910 deg | 0.226 rad | Reviewed 55 deg / 0.55 rad pass |
| Galbot Golf | `105` | Video2Data | 0.700 mm | 49.157 deg | 0.520 rad | Reviewed 55 deg / 0.55 rad pass |
| Galbot Golf | `105` | Phantom | 0.645 mm | 37.143 deg | 0.391 rad | Reviewed 55 deg / 0.55 rad pass |
| Galbot Golf | `253` | Ground truth | 0.229 mm | 14.368 deg | 0.368 rad | Strict pass |
| Galbot Golf | `253` | Video2Data | 0.244 mm | 16.497 deg | 0.366 rad | Strict pass |
| Galbot Golf | `253` | Phantom | 0.250 mm | 18.014 deg | 0.366 rad | Strict pass |
| YAM | `105` | Ground truth | 0.232 mm | 7.883 deg | 0.138 rad | Strict pass |
| YAM | `105` | Video2Data | 0.249 mm | 14.321 deg | 0.369 rad | Strict pass |
| YAM | `105` | Phantom | 0.242 mm | 6.872 deg | 0.171 rad | Strict pass |
| YAM | `253` | Ground truth | 0.179 mm | 6.164 deg | 0.227 rad | Strict pass |
| YAM | `253` | Video2Data | 0.135 mm | 9.870 deg | 0.362 rad | Strict pass |
| YAM | `253` | Phantom | 0.479 mm | 11.497 deg | 0.288 rad | Strict pass |

YAM's strict result requires an embodiment-level calibration, not a relaxed
solver: a contact-equivalent left-jaw `Rz(pi)` mapping, right-jaw identity, and
a fixed `+0.15` m forward / `-10` degree roll base alignment. This removes the
original left-wrist branch flips while remaining fixed across all trackers and
both clips.

All four five-panel grids contain exact source-length synchronized streams at
30 FPS: source, E2FGVI arms-masked, GT-driven robot, Video2Data-driven robot,
and Phantom-driven robot. Renders and composites are `1920x1080`; review grids
are `1920x720`. The production planner validates all 28 render, composite, and
grid stages as `skipped_complete`, including output hashes and lineage. Visual
QA at start/middle/end and the Galbot-105 stress frames found no corrupt
panels, detached meshes, joint branch flips, or obvious object-depth ordering
errors. Galbot's forearms remain large near the image borders. YAM retains the
predominantly black and gray materials authored in its USD.

## GraspGenX contact refinement

The parallel-jaw limitation was tested with the official GraspGenX v1.0.0
implementation at commit `b9429097728cb1c430dd78b92edf17ba318aad03` and its
release generator/discriminator checkpoints. For every reconstructed object
and robot profile, GraspGenX sampled 600 sweep-volume-conditioned grasps from
the metric SAM3D mesh; the 150 highest-confidence candidates were retained.
Clip `253` required the same RGB-only MoGe, SAM2, SAM3D, and FoundationPose
object-reconstruction stages already used for `105`.

Phantom aperture closure supplies a temporal proposal, but RGB-visible contact
is the final gate because Phantom closes before contact on `105`, spuriously
starts the cup event at frame 0 on `253`, and the brush is already grasped when
`253` begins. The final inclusive interaction windows are board
`26/30/124`, knife `22/30/132`, cup `8/14/58`, and brush `0/0/63`, expressed
as start/anchor/end frames.

At each anchor, the V2D thumb/index contact pair is expressed in the
FoundationPose object frame. A single midpoint-to-mesh translation preserves
the observed pair's aperture and direction; projecting the two tips
independently was rejected because it collapsed both contacts onto one nearby
surface patch. Candidate ranking uses this registered contact-pair residual as
the primary term, then weaker confidence, human-pose, and approach
terms. Parallel-jaw `Rz(pi)` symmetry is resolved against the human pose.
Finally, the anchor transform is propagated as one constant right-multiplied
base-local offset through the approach, hold, and release intervals with
C2/Slerp boundary blending. This avoids inheriting per-frame FoundationPose
jitter.

| Robot | Clip | Interaction | Selected aperture | Grasp confidence | Pair residual | Human-orientation delta |
|---|---|---|---:|---:|---:|---:|
| Galbot Golf | `105` | left / board | 4.1 mm | 0.957 | 13.5 mm | 17.9 deg |
| Galbot Golf | `105` | right / knife | 19.2 mm | 0.987 | 10.7 mm | 18.3 deg |
| YAM | `105` | left / board | 4.3 mm | 0.954 | 13.3 mm | 11.2 deg |
| YAM | `105` | right / knife | 20.6 mm | 0.981 | 9.4 mm | 22.2 deg |
| Galbot Golf | `253` | left / cup | 4.3 mm | 0.893 | 4.3 mm | 36.7 deg |
| Galbot Golf | `253` | right / brush | 35.7 mm | 0.929 | 17.7 mm | 58.0 deg |
| YAM | `253` | left / cup | 8.8 mm | 0.853 | 2.4 mm | 15.4 deg |
| YAM | `253` | right / brush | 29.2 mm | 0.886 | 20.7 mm | 56.1 deg |

The brush is the weakest interaction: both embodiments retain roughly
`18--21` mm contact residual and `56--58` degree separation from the human
orientation prior.

All four refined trajectories render and composite at exact source length.
The production IK residuals remain sub-millimeter, although transient
trajectory frames require disclosed gates beyond the contact-frame previews:

| Robot | Clip | Max position residual | Max orientation residual | Max arm step | Production gate |
|---|---|---:|---:|---:|---|
| Galbot Golf | `105` | 0.655 mm | 61.924 deg | 0.572 rad | 65 deg / 0.58 rad |
| YAM | `105` | 0.247 mm | 23.135 deg | 0.395 rad | 25 deg / 0.45 rad |
| Galbot Golf | `253` | 0.257 mm | 16.649 deg | 0.452 rad | 22 deg / 0.50 rad |
| YAM | `253` | 0.140 mm | 24.603 deg | 0.432 rad | 26 deg / 0.45 rad |

The result is promising locally but not yet a metric grasp-recovery result.
The common translation needed to reconcile V2D hand contacts with the
reconstructed object mesh is `172.6--305.5` mm across the eight interactions,
far larger than the final `2.4--20.7` mm registered pair residuals. The
four-panel videos therefore show whether GraspGenX improves jaw geometry and
orientation after registration; they do not establish that independently
reconstructed hand and object depth are globally aligned. Joint hand/object
scale and depth calibration is the next prerequisite for a physical success
metric. Grasp generation and selection are RGB/V2D-only, but the final
comparison overlays intentionally retain the same TACO GT object-depth
occluder as the baseline to isolate the trajectory change; the visualization
is therefore not a pure-RGB end-to-end composite.

### Contact-wrench reranking

The follow-up adds a deterministic, simulator-free port of the contact-wrench
geometry used by Video2Data's CHORD reward. Each candidate uses its two
re-derived mesh contacts, outward mesh normals converted to inward contact
forces, `mu=0.1`, eight friction-cone edges, and object-radius-normalized torque.
A shared 512-direction 6D unit basis generated with PCG64 seed 0 makes scores
comparable across candidates. The objective rewards the tenth percentile of
the support envelope (`q10`) at weight `1.0`, while retaining registered
contact residual at weight `5.0` and confidence, pose translation, pose
rotation, and approach priors at `0.005`, `0.1`, `0.04`, and `0.015`.
Registration magnitude remains unpenalized. No exact reference wrench envelope
or simulator rollout was used.

The table separates the unconstrained wrench ranking from the pose ultimately
rendered under the unchanged embodiment IK gates. `q10` is dimensionless
because torque is normalized by the reconstructed object's bounding radius.

| Robot | Clip | Interaction | Contact-selected candidate (`q10`) | Wrench-ranked candidate (`q10`) | Rendered strict-feasible candidate / aperture (`q10`) | Rendered `q10` change |
|---|---|---|---:|---:|---:|---:|
| Galbot Golf | `105` | left / board | 130 (0.1548) | 130 (0.1548) | 130 / 4.15 mm (0.1548) | 0.0% |
| Galbot Golf | `105` | right / knife | 101 (0.1088) | 101 (0.1088) | 101 / 19.20 mm (0.1088) | 0.0% |
| YAM | `105` | left / board | 126 (0.1239) | 43 (0.1768) | 126 / 4.33 mm (0.1239) | 0.0% |
| YAM | `105` | right / knife | 89 (0.1167) | 89 (0.1167) | 89 / 20.57 mm (0.1167) | 0.0% |
| Galbot Golf | `253` | left / cup | 13 (0.1171) | 18 (0.1346) | 18 / 9.09 mm (0.1346) | +15.0% |
| Galbot Golf | `253` | right / brush | 23 (0.1316) | 23 (0.1316) | 23 / 35.73 mm (0.1316) | 0.0% |
| YAM | `253` | left / cup | 85 (0.1139) | 85 (0.1139) | 85 / 8.79 mm (0.1139) | 0.0% |
| YAM | `253` | right / brush | 81 (0.1151) | 77 (0.1376) | 77 / 32.73 mm (0.1376) | +19.5% |

The unconstrained YAM-105 board winner, candidate 43, would improve `q10` by
42.7%, but reaches `34.842927` degrees against the unchanged 25-degree gate.
The next two ranked candidates, 145 and 50, also fail at `28.743892` and
`37.294588` degrees. Rank-four candidate 126 passes at `23.135286` degrees and
is retained as the best strict-IK-feasible fallback. This external feasibility
pass is material: the static contact-wrench score alone does not encode the
embodiment's full-arm reachability.

The midpoint registration is applied as a constant world correction to the
estimated object pose, leaving the selected object-to-gripper transform
mesh-valid at the anchor. Contacts are re-derived after symmetry selection and
then revalidated. The trajectory uses the same anchor-derived
`base_local_offset` across the interaction, so mesh contact is guaranteed only
at the anchor, not throughout the hold. The required registration remains
large (`175.8--305.5` mm), so wrench reranking does not resolve the upstream
hand/object metric-alignment limitation.

All three visual conditions were rerendered under the same immutable
`robotic-grounding:photo-render-v8` image
(`sha256:09f0bb3becf4c6ee16b701b049254c384df35c97dd1d22e403da0f7d2f7c2f1b`):
V2D baseline, contact-selected GraspGenX, and wrench-reranked GraspGenX. The
position, orientation, joint-step, and orientation-cost gates were unchanged
between methods.

| Robot | Clip | Wrench max position residual | Wrench max orientation residual | Wrench max arm step | Gate result |
|---|---|---:|---:|---:|---|
| Galbot Golf | `105` | 0.655 mm | 61.924 deg | 0.572 rad | 65 deg / 0.58 rad pass |
| YAM | `105` | 0.247 mm | 23.135 deg | 0.395 rad | 25 deg / 0.45 rad pass |
| Galbot Golf | `253` | 0.260 mm | 5.617 deg | 0.452 rad | 22 deg / 0.50 rad pass |
| YAM | `253` | 0.191 mm | 16.489 deg | 0.405 rad | 26 deg / 0.45 rad pass |

## Object compositing on the 105 clip

This three-way ablation changes only how the knife and cutting-board/plate
occluder is constructed. All conditions use the same E2FGVI base video, the
same Video2Data robot RGB/mask/depth render, and the same 3 mm depth guard:

1. TACO GT meshes and per-frame object poses.
2. RGB-only SAM2 object masks with per-frame MoGe metric depth.
3. Video2Data object reconstruction: SAM3D meshes and FoundationPose poses,
   rendered into metric camera-z depth.

Conditions 2 and 3 have RGB-only upstream inputs. Two human-provided bounding
boxes on RGB frame 0 initialize the knife and board SAM2 tracks; they are not
GT object masks. MoGe estimates depth and camera intrinsics from RGB, and the
third condition uses the pipeline's canonical smoothed FoundationPose tracks
as its primary result. Neither estimated condition consumes TACO object
meshes, object poses, masks, depth, or camera calibration. GT object data is
used only by condition 1 and by the post-hoc evaluator.

| Occluder condition | Robot-visible IoU | Wrong robot-depth decisions | Object-mask IoU | Overlap depth MAE | False visible | False occluded | Temporal disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT mesh + GT pose | 1.0 | 0 (0%) | 1.0 | 0 m | 0 | 0 | 0% |
| RGB masks + MoGe depth | 0.9759203532 | 1,325,469 (2.38893297%) | 0.6858006999 | 0.1762251283 m | 1,190,276 (73.0769% of GT-occluded pixels) | 135,193 | 0.998360% |
| SAM3D mesh + FoundationPose pose (smoothed) | 0.9798924473 | 1,100,569 (1.98358888%) | 0.7109262537 | 0.1752361543 m | 879,185 (53.9775% of GT-occluded pixels) | 221,384 | 1.089252% |

The reconstructed-mesh condition improves the primary visibility metric and
reduces total decision errors by 224,900 relative to dense RGB compositing. It
also reduces false-visible errors by 311,091, although false-occluded errors
and temporal disagreement increase. Both estimated methods still have about
17.5 cm overlap depth error, so this comparison supports estimated-depth
compositing while also exposing metric depth/pose as the remaining bottleneck.
As a sensitivity check, rendering the unsmoothed FoundationPose tracks gives
0.979766 visible IoU, 1,107,596 decision errors, and 1.165917% temporal
disagreement. The canonical smoothed result is therefore retained as the
primary condition despite localized knife-alignment regressions in its pose QA.

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

The paths in this section are local, git-ignored study outputs. They are not
included in the pushed source branch.

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
- Sequence-`105` three-way object-compositing comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/object_compositing_v1/object_compositing_3way_105.mp4`
- Start/middle/end and error-focused object-compositing review sheets:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/object_compositing_v1/object_compositing_3way_105_{contact,error_frames}.jpg`
- Verified GT, RGB-only, and Video2Data object-compositing evaluations:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/object_compositing_v1/{ground_truth,rgb_estimated_depth_verified,v2d_estimated_object}/evaluation_vs_gt.json`
- Galbot Golf five-panel `105` comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/parallel_jaw/galbot_one_golf/final_5panel_comparison.mp4`
- Galbot Golf five-panel `253` comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/parallel_jaw/galbot_one_golf/final_5panel_comparison.mp4`
- YAM five-panel `105` comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/parallel_jaw/yam_bimanual/final_5panel_comparison.mp4`
- YAM five-panel `253` comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/parallel_jaw/yam_bimanual/final_5panel_comparison.mp4`
- Parallel-jaw QA sheets:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/parallel_jaw_qa/`
- GraspGenX four-panel comparisons (source, E2FGVI, V2D baseline, refined V2D):
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/parallel_jaw/<robot>/graspgenx_v2d_4panel_comparison_<clip>.mp4`
- GraspGenX selected targets and per-interaction provenance:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/parallel_jaw/graspgenx_targets/<robot>/v2d_graspgenx_aligned/`
- GraspGenX start/contact/end QA sheets:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/graspgenx_parallel_jaw/`
- V8 three-method grasp comparisons (V2D baseline, contact-selected GraspGenX,
  contact-wrench-reranked GraspGenX):
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/parallel_jaw/<robot>/v8_v2d_grasp_methods_3panel_<clip>.mp4`
- Contact-wrench GraspGenX targets and per-interaction score provenance:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/parallel_jaw/graspgenx_targets/<robot>/v2d_graspgenx_wrench_v1/`

The sequence-`105` RGB-only and reconstructed-mesh object-compositing ablation
and the `105`/`253` GraspGenX parallel-jaw and contact-wrench ablations are now
complete. The next useful step is a joint hand/object metric-alignment stage
before optimizing time-varying contact consistency or evaluating grasp
stability.
