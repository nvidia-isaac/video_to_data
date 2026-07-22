# Visual Inpainting Investigation Tracker

## Objective

Evaluate hand-tracking inputs for a Phantom-style E2FGVI human-removal and
robot-overlay pipeline on small TACO segments:

1. Phantom tracking
2. Video2Data reconstruction tracking
3. TACO ground-truth motion tracking

The initial robot target is Dexmate Vega with Sharpa hands. All production
stages should remain container-compatible for later OSMO execution.

## Status

- [x] Read the project brief.
- [x] Create a dedicated git branch.
- [x] Map the Phantom pipeline and artifacts.
- [x] Map Video2Data reconstruction, retargeting, and rendering interfaces.
- [x] Inventory available TACO RGB clips and motion-tracking ground truth.
- [x] Identify a small, representative demo set.
- [x] Build the initial experiment harness, contracts, and resolved manifest.
- [x] Containerize and validate the E2FGVI backend.
- [x] Implement tracker adapters and shared arm-mask generation.
- [x] Implement calibrated Vega + Sharpa rendering and compositing.
- [x] Run Phantom-tracking + E2FGVI + robot-overlay demos.
- [x] Run Video2Data-tracking + E2FGVI + robot-overlay demos.
- [x] Run ground-truth-tracking + E2FGVI + robot-overlay demos.
- [x] Render side-by-side comparisons and summarize findings.

## Working Log

### 2026-07-22

- Started from clean `main` at `8db0c9b1`.
- Created and finalized branch
  `mverghese/visual-inpainting-investigation` using the repository's
  `<username>/<topic>` convention.
- Began parallel repository, dataset, robot-asset, and environment audits.
- Confirmed Phantom's reusable E2FGVI seam is strictly RGB frames plus an
  `(N,H,W)` binary arm mask. Phantom's robot twin is not reusable for Sharpa:
  it supports scalar Robotiq aperture and a fixed non-TACO calibration.
- Confirmed all Phantom submodules and model checkpoints are absent locally.
- Inventoried 2,311 TACO RGB clips (all 1920x1080, 30 FPS) and 80 processed
  Sharpa trajectories in the sibling data checkout. Only sequence
  `taco_empty__kettle__plate_20231031_060` has its original raw MANO/object GT
  locally.
- Selected sequences `060`, `105`, and `253` for GT uniqueness, an existing
  high-quality robot-render reference, and short turnaround respectively.
- Added versioned tracking/mask/render contracts, a strict source resolver,
  renderer-neutral compositor, comparison-grid renderer, and CPU tests.
- Resolved all selected RGB and motion inputs with exact frame-count matches.
- Audited Video2Data's WiLoR/HaMeR result bundle, historical Sharpa adapter,
  current retargeting schema, and photorealistic replay branch. The reusable
  adapter needs validity/scale/FPS fixes; its current reconstruction images are
  not Blackwell-compatible.
- Added a strict TACO GT adapter and exported frame-aligned tracking and
  Vega/Sharpa trajectory archives for all three selected sequences.
- Upgraded the SAM2 image to PyTorch 2.7.1/CUDA 12.8, pinned SAM2 to commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`, and verified CUDA execution on
  the RTX PRO 6000 Blackwell host.
- Generated and visually checked two-arm SAM2 masks for all 381 selected
  frames. Per-object largest-component cleanup removes isolated prompt
  speckles before the strict boolean mask archive is written.
- Added and built the pinned E2FGVI-HQ stage at Phantom release commit
  `5b45ffe400288006facb350e00d319bfc6c5cbd3`. The exact 164.5 MB checkpoint
  loads strictly as weights-only; inference is offline and its input mounts are
  read-only.
- Completed E2FGVI human removal for all three clips with one shared config:
  960-pixel processing cap, four 3x3 cross-kernel dilation passes,
  `neighbor_stride=5`, `ref_stride=20`, and `num_ref=5`. Every output re-probed
  at 1920x1080, 30 FPS, with exact source frame count.
- Visual QA confirms the knife, kettle, plate, cup, and brush remain present.
  E2FGVI leaves some original cast shadows and mild texture smearing; these are
  recorded as baseline limitations rather than silently post-processed.
- Downloaded the official public TACO egocentric camera package for the three
  selected sequences and validated its intrinsics and per-frame
  world-to-camera transforms against the source geometry and frame counts.
- Projected the GT MANO skeletons into every source clip. Visual QA at the
  beginning, middle, and end of each sequence shows the projected skeletons
  closely following the recorded hands, validating the camera convention,
  frame alignment, scale, and world-coordinate interpretation used by the
  renderer.
- Added an offline Vega + articulated Sharpa renderer with strict trajectory,
  calibration, URDF, joint-limit, IK, visibility, and output validation. Its
  real-asset dry run succeeds in the existing pinned photo-render container.
- Completed calibrated full-resolution GT Vega + Sharpa renders for all three
  clips. Every robot is visible in every frame, every GT wrist projects inside
  its matching image, max IK attachment residual is 0.063--0.524 mm, and max
  arm-joint step is 0.156--0.309 rad/frame.
- Rendered source/E2FGVI/robot and no-inpainting comparison videos for all
  three clips. Visual QA confirms that the Sharpa hands follow the original
  human wrists and grasps; the hard-overlay prototype still draws some robot
  pixels in front of manipulated objects.
- Verified object-depth compositing is feasible from the processed TACO
  object poses and meshes. On sequence 253, 27.4% of robot/object overlap
  pixels place the real object closer than the robot, so depth-aware occlusion
  was promoted from a future improvement into the initial GT result.
- Completed strict object mask/depth bundles and depth-aware final composites
  for all three clips with a 3 mm depth guard. The compositor suppresses
  374,478 robot pixels behind the kettle/plate, 147,752 behind the knife/plate,
  and 271,683 behind the brush/cup. It validates committed robot/object bundle
  metadata, rejects path aliases, fingerprints every input/output, and retains
  hard-mask compositing only as an explicit diagnostic fallback.
- Added a resumable GT batch orchestrator with a read-only plan default,
  mandatory render/object-depth/composite/grid ordering, explicit GPU and
  overwrite authorization, and strict bundle validation. The live resolved
  manifest now reports all 12 sequence-stage actions as `skipped_complete`.
- Validated the supplied licensed MANO v1.2 left/right assets by exact SHA-256
  and used them only through read-only container mounts or ignored runtime
  weight directories.
- Built a Blackwell-compatible, pinned WiLoR image and completed offline,
  bimanual Video2Data inference for all three clips. Valid tracking counts are
  `155/146`, `152/152`, and `74/74` for left/right on clips 060, 105, and 253.
  The nine-frame right-hand detection gap on 060 remains explicitly invalid;
  it was not interpolated or hidden.
- Split learned tracking conversion into auditable MANO forward-kinematics and
  Sharpa-retargeting stages. The definitive `253` rerun records exact source
  video, camera, MANO, public-weight, source-revision, immutable container,
  implementation-source, and complete Sharpa XML/mesh fingerprints.
  The explicit 70 mm Sharpa task-space convergence gate accepts all valid
  Video2Data frames; a tighter 50 mm diagnostic rejected real learned poses
  and is retained as a documented non-convergence result.
- Completed and visually checked the full 74-frame Video2Data 253 condition:
  the shared E2FGVI video, calibrated Vega + Sharpa render, and fixed TACO
  object-depth composite all pass strict validation. The learned arm trajectory
  requires a documented 12 mm Vega attachment-residual gate (10.94 mm observed
  maximum) and remains below the 0.4 rad/frame joint-step gate.
- Containerized Phantom's Grounding DINO + HaMeR path with pinned source/model
  revisions, weights-only checkpoint loading, real TACO intrinsics, offline
  inference, and deterministic bimanual identity assignment. Raw tracking and
  visual overlays are complete for all three clips. A MANO joint-frame
  convention error was caught at the Sharpa boundary before rendering; the
  corrected native-side `AxisLayerFK` conversion agrees with manotorch to
  `4.9e-7` maximum rotation-matrix error and now accepts every Phantom
  observation at the declared Sharpa gate. Phantom validity is `155/148`,
  `152/152`, and `74/74`; clip 060's seven absent source observations remain
  invalid and were not interpolated.
- Completed and visually checked a second full Video2Data condition on the
  152-frame 105 clip. Its Vega solver stays below 0.20 mm attachment residual
  and 0.390 rad/frame; the depth-aware composite hides 1,628,800 robot pixels
  behind the real knife and plate.
- Completed and visually checked the full 74-frame Phantom 253 condition. The
  common 12 mm Vega attachment gate first failed explicitly at a 24.71 mm
  maximum; a reviewed 30 mm condition gate then completed with a 0.277 mm p95
  residual, 0.346 rad/frame maximum joint step, and all 74 robot frames
  visible. The final depth composite hides 548,627 robot pixels behind the
  real brush and cup. The single worst-case IK outlier remains a reported
  Phantom limitation rather than being averaged away.
- Regenerated the definitive `253` WiLoR observation directory as one atomic
  74-frame generation with a strict `run_generation.json` commit marker. A
  recursive semantic comparison against the legacy JSON covered 11,766 parsed
  nodes and found zero differences (`max_abs_diff=0.0`); byte changes are only
  sorted keys plus a trailing newline. The no-model strict-resume path then
  revalidated the exact video, weights, image, sources, frame set, and output
  hashes.
- Upgraded both `253` learned conditions to tracking/Sharpa v2 sidecars. The
  Video2Data and Phantom robot-trajectory arrays are exactly unchanged; the
  Phantom trajectory is also byte-identical. Both Sharpa runs accepted 74/74
  frames per side and bind 18 implementation sources plus all 46 consumed
  robot XML/mesh files. Prior directories are preserved under explicit
  `tracking_legacy_pre_provenance_v2` names.
- Re-rendered Video2Data `253` against the current renderer sources and new
  committed trajectory hash. Its strict learned-condition plan now reports all
  three stages `skipped_complete`; Phantom also remains strictly complete at
  its reviewed 30 mm Vega gate because its trajectory bytes did not change.
- Refreshed deterministic tracking evaluation and created the final five-way
  `253` comparison. Visual QA of frames 0, 37, and 73 confirms synchronized
  source/E2FGVI/GT/Video2Data/Phantom geometry, coherent contact placement, and
  expected depth-aware object occlusion. Video2Data has lower metric 3D error
  on both hands; Phantom is slightly better in left-hand projected 2D error.
- Hardened E2FGVI publication against interrupted overwrites: a pre-inference
  `committing` state can no longer be mistaken for a complete artifact, and a
  completed run binds input/output bytes and SHA-256. Legacy sidecars can be
  enriched only with facts observable now and are explicitly marked
  `legacy_unrecorded` when their historical image identity is unavailable.
- Rebuilt E2FGVI at Phantom commit
  `5b45ffe400288006facb350e00d319bfc6c5cbd3` as immutable image
  `sha256:398b54800eebd0343ec27ba86c1a59829cb7439ce9418ec533744e837558ebbc`
  and reran clip `253` on GPU 0. Its output remained byte-identical at SHA-256
  `250a7e7a5a1de96995b2d32d68a56b4dfac1413a492415237e695982b1ad6c08`.
  The `060` and `105` outputs were safely enriched with observed output hashes
  while retaining explicit unknown historical image identity.
- Reran all three TACO object-depth bundles with immutable renderer image
  `sha256:86ca30a0310c25fa3c0eb5e28a282b21e8a95b151e7011a00bbe4f3bbc06ed63`
  and complete source/input fingerprints. Every object mask and depth array was
  byte-identical to the previous reviewed generation. Republished all GT
  composites/grids and both definitive `253` learned composites/grids against
  the refreshed object metadata.
- Revalidated strict resume after publication: all 12 GT actions report
  `skipped_complete`, as do all three Video2Data `253` actions at the 12 mm
  Vega gate and all three Phantom `253` actions at its reviewed 30 mm gate.
  Rebuilt the final five-way and all-sequence audit videos and repeated visual
  QA at frames 0, 37, and 73.
- An independent compatibility review caught two pre-commit isolation defects:
  ordinary input files exposed their whole parent directory, which rejected
  the documented nested `data/clip.mp4` to `data/outputs/clip` run layout, and
  WiLoR's atomic mount granted write access to sibling run artifacts. The final
  wrapper bind-mounts declared files exactly and read-only, retains one legacy
  alias for non-strict exact shared roots so HOI relative symlinks still work,
  and confines WiLoR writes to a private same-filesystem staging parent before
  host-side atomic publication. Focused tests cover nested SAM2/WiLoR runs,
  committed read-only resume, incomplete-output refusal, cleanup, symlink
  compatibility, and mount confinement; an actual SAM2 Docker file-mount smoke
  test also passed.
- A final API review also caught SAM2's historical any-file completion check
  and a positional-argument regression in the hardened WiLoR wrapper. SAM2 now
  publishes a marker-last atomic generation with immutable image, input,
  implementation, exact frame-set, and output-hash validation; committed
  generations are revalidated read-only and partial directories fail closed.
  WiLoR retains its original positional `bboxes_dir` and `dev` slots, with new
  image/GPU controls keyword-only. Pre-manifest partial outputs no longer
  false-skip in either stage.
- Final local verification passes 225 tests plus 14 parameterized subtests,
  scoped Ruff checks, Python byte-compilation, and Git whitespace validation.
  A pre-commit audit found no credentials, licensed/model data, media, or
  generated caches in the source set; the 26 GB runtime artifact tree remains
  ignored and the licensed MANO directory remains outside the Git worktree.

## Decisions

- Keep all new implementation under top-level `inpainting/`.
- Treat rendered video artifacts as required validation outputs, not optional
  post-processing.
- Do not interrupt GPU processes owned by other projects.
- Use one identical E2FGVI configuration for every tracking condition.
- Require boolean mask archives at the E2FGVI seam and deserialize checkpoints
  only with PyTorch's `weights_only=True` mode.
- Keep Vega/Sharpa IK + rendering behind a renderer-neutral RGB/mask/depth
  contract rather than importing Phantom's Panda/Kinova MuJoCo twin.
- Treat the sibling `/home/mverghese/video_to_data_internal` checkout as a
  read-only data/asset source; all implementation and versioned changes remain
  in this repository.

## Open Questions / Risks

- E2FGVI-HQ, SAM2, WiLoR, HaMeR, Grounding DINO, and licensed MANO assets are
  now local and fingerprinted. Learned inference is offline after acquisition.
- The working clone's tracked Sharpa meshes are unsmudged LFS pointers and its
  branch predates Vega. Complete Vega/Sharpa assets are available in the sibling
  checkout and on `origin/hot3d-photorealistic-rendering`.
- Disk space was restored externally and is sufficient for the remaining demo
  renders. Build/download stages remain selective; unrelated Docker cache is
  never pruned.
- Phantom's upstream non-EPIC DINO code cannot assign bimanual
  (`target_hand=both`) detections; the local adapter provides deterministic
  image-side/anatomical-side assignment and reports ambiguity instead of
  silently swapping identities.
- The `060` and `105` E2FGVI outputs predate immutable-image recording. Their
  current sidecars bind the exact observed outputs but intentionally retain
  `container_image_provenance=legacy_unrecorded`; only the definitive `253`
  E2FGVI generation has a rerun-verified immutable image identity.
- The definitive `253` v2 tracking/Sharpa sidecars preserve `/repo/...` paths
  from their atomic candidate-generation workspace. That directory was
  published under the canonical `tracking` name afterward. The current files
  match the recorded hashes and the learned renderer validates the actual
  trajectory content, but those recorded paths describe the generation
  workspace rather than live host locations.

## Artifact Index

- TACO candidate contact sheet:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/candidate_contact_sheet.jpg`
- Synchronized selected-demo RGB video grid:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_demo_video_grid.mp4`
- Resolved input manifest:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json`
- GT tracking and Vega/Sharpa world-trajectory diagnostics:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/ground_truth/tracking/`
- Official TACO camera parameters:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/source_data/Egocentric_Camera_Parameters/`
- GT hand-camera alignment previews:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/ground_truth/tracking/camera_overlay.mp4`
- Shared SAM2 masks and full-resolution mask previews:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/shared_arm_mask/`
- E2FGVI human-removal videos and deterministic metadata:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/shared_inpaint/e2fgvi_960.{mp4,json}`
- Nine-panel source/mask/E2FGVI comparison (74 synchronized frames):
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_shared_mask_e2fgvi_grid.mp4`
- Per-sequence source/E2FGVI/GT Vega+Sharpa comparisons:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_{060,105,253}_gt_final_grid.mp4`
- Final depth-aware per-sequence comparisons:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_{060,105,253}_gt_depth_final_grid.mp4`
- Resumable-batch per-sequence comparison bundles:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/ground_truth/final_comparison_grid.{mp4,json}`
- Final synchronized nine-panel GT comparison (74 frames):
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_gt_depth_overlay_all_sequences.mp4`
- Hard-overlay nine-panel audit baseline:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_gt_hard_overlay_all_sequences.mp4`
- Per-sequence source/no-inpainting/E2FGVI robot comparisons:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_{060,105,253}_gt_inpainting_comparison.mp4`
- GT object-projection calibration sheets:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/object_projection_{060,105,253}.jpg`
- Video2Data learned-tracking camera overlays:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/v2d/audit/tracking_overlay.mp4`
- Video2Data 253 full learned-condition comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/v2d/final_comparison_grid.mp4`
- Phantom bimanual hand-tracking overlays:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/<sequence>/phantom/tracking/hand_overlay.mp4`
- Phantom 253 full learned-condition comparison:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/phantom/final_comparison_grid.mp4`
- Final five-way 253 tracker comparison and start/middle/end contact sheet:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_253_tracker_comparison.{mp4,_contact.jpg}`
- Deterministic learned-tracker evaluation reports:
  `/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_dust__brush__cup_20231005_253/{v2d,phantom}/tracking/evaluation_vs_ground_truth.json`
