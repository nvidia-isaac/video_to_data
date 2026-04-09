# Unified 4D Gaussian Splatting for Holistic Scene Reconstruction from Monocular Video

## 1. Problem Statement

Given a monocular RGB video of a scene containing people, hands, and unknown rigid objects, jointly reconstruct:

1. **Static scene geometry** (room, furniture, floor)
2. **Articulated human bodies** (full-body pose and shape over time)
3. **Hands** (high-DOF articulation, fine-grained geometry) — optional
4. **Unknown rigid objects** (no prior template or CAD model)
5. **Camera trajectory** (ego-motion) — solved jointly or provided externally

...all in a shared coordinate system, with enough semantic structure to recover parametric model fits (SMPL, SMPL+H, or SMPL-X) as a byproduct of reconstruction. Metric scale can be recovered post-hoc if needed.

## 2. Core Idea

Represent the entire dynamic scene as a set of **semantically-typed 4D Gaussians**. Each Gaussian carries:

- Standard 3DGS attributes (position, rotation, scale, opacity, spherical harmonics)
- An **entity type** (background / body / hand / object)
- **Entity-specific motion parameters** (LBS skinning weights for body/hand, SE(3) rigid transform for objects, identity for background)

Deformation over time is governed by the entity type: body Gaussians move via SMPL forward kinematics, hand Gaussians via MANO (if provided), object Gaussians via per-frame rigid transforms, and background Gaussians remain static. All parameters — including body pose, object poses, Gaussian attributes, and camera parameters — are jointly optimized through differentiable rendering against the observed video frames.

Contact between hands and objects is **implicit**: it emerges from the shared photometric optimization rather than being enforced through explicit penetration/attraction losses.

## 3. Per-Gaussian Representation

```
StandardAttributes:
    position        : Float[3]       # canonical-space center
    rotation        : Float[4]       # quaternion
    scale           : Float[3]       # log-scale per axis
    opacity         : Float[1]       # sigmoid -> [0,1]
    color_sh        : Float[K]       # spherical harmonics (K=48 for degree 3)

EntityAttributes:
    entity_id       : Int            # 0=background, 1=body, 2=left_hand, 3=right_hand, 4+=objects
                                     # fixed at initialization, inherited on clone/split

BodyAttributes (entity_id == 1):
    smpl_vertex_id  : Int            # nearest body model vertex
    skinning_weights: Float[J_body]  # blend weights for body joints

HandAttributes (entity_id == 2 or 3):  # optional
    mano_vertex_id  : Int            # nearest MANO vertex
    skinning_weights: Float[J_hand]  # blend weights for MANO joints

ObjectAttributes (entity_id >= 4):
    rigid_body_id   : Int            # which rigid body this belongs to
```

Entity decomposition does not require per-Gaussian learned semantic features. The `entity_id` is structurally enforced: it is set at initialization from SAM2 masks, inherited during densification, and reinforced by the fact that each entity type uses a different deformation model (LBS vs. SE(3) vs. static). The `L_mask` loss against SAM2 per-frame segmentations provides sufficient supervision for entity boundaries.

## 4. Parametric Body Model

The module is agnostic to which body model is provided. The deformation model is parameterized by whichever model is available:

| Model | Joints | Use case |
|-------|--------|----------|
| SMPL | 24 | Body only, simplest, supported by `v2d_nlf` |
| SMPL+H | 51 | Body + hands, still supported by `v2d_nlf` |
| SMPL-X | 55 | Body + hands + face, requires a separate estimator module |

**Default / starting point: SMPL body only**, sourced from `v2d_nlf`. Hand entities (entity_id 2 and 3) are simply absent until a MANO estimator is wired in. The optimization gracefully degrades — hands are treated as unlabelled background or object Gaussians rather than failing.

## 5. Deformation Model

Each Gaussian is transformed from canonical space to observation space at frame `t`:

**Body Gaussians:** Forward linear blend skinning (LBS) using joint transforms derived from per-frame pose parameters `theta_t` and shared shape parameters `beta`:

```
T_body(x, t) = sum_j w_j * G_j(theta_t, beta) * x
```

where `w_j` are the per-Gaussian skinning weights and `G_j` are joint transforms from forward kinematics.

**Hand Gaussians (optional):** Same LBS mechanism using MANO joint transforms from per-frame hand pose `phi_t` and hand shape `psi`:

```
T_hand(x, t) = sum_j w_j * H_j(phi_t, psi) * x
```

**Object Gaussians:** Per-frame SE(3) rigid body transform:

```
T_object(x, t) = R_t * x + t_t
```

where `(R_t, t_t)` are per-frame rotation and translation for each rigid body.

**Background Gaussians:** Identity (static).

An optional **non-rigid correction MLP** can be added per entity type to capture pose-dependent deformations (clothing wrinkles, soft-tissue dynamics) that LBS cannot model:

```
T_final(x, t) = T_entity(x, t) + delta_MLP(x, pose_t)
```

## 6. Initialization Pipeline

All initialization is performed once before optimization, primarily from **frame 0**. Monocular depth defines the canonical coordinate system for the entire scene; all other components (body model, SAM 3D meshes, camera poses) are aligned into depth-space.

### 6.1 Coordinate System: Depth as Common Reference

Monocular depth serves as both the **initialization coordinate system** and the **regularization signal** during optimization. Using depth as the shared reference avoids scale/shift mismatches between the depth regularizer and the scene geometry — everything lives in the same space by construction.

Any monocular depth model can be used (MoGe, UniDepth, Depth Anything, etc.) provided it outputs temporally consistent depth maps. The module consumes depth as pre-computed uint16 PNG files and intrinsics as JSON — the source model is irrelevant.

1. Run a monocular depth model on the full video to obtain depth maps `D_t` for all frames
2. Frame 0's depth map `D_0` with camera intrinsics `(fx, fy, cx, cy)` defines world coordinates:
   ```
   For pixel (u, v) with depth d = D_0(u, v):
       X = (u - cx) * d / fx
       Y = (v - cy) * d / fy
       Z = d
   ```
3. This coordinate system is **not necessarily metric** (relative depth models output consistent but unscaled depth). Metric scale can be recovered post-hoc if needed by comparing the optimized body height against expected human dimensions (~1.7m). Metric depth models (UniDepth, MoGe) produce a metric coordinate system directly.

### 6.2 Aligning Entities to Depth-Space

All entity initializations are aligned to depth-space. This ensures zero tension between initialization and the `L_depth` regularization loss.

**Body model (SMPL / SMPL+H / SMPL-X):**
1. The body estimator (e.g., `v2d_nlf` `run_align_nlf_to_depth`) outputs body parameters already aligned to depth-space
2. No additional alignment step is required if `run_align_nlf_to_depth` has been run

**SAM 3D Objects (per detected object):**
- Scale-align the SAM 3D mesh to depth-space by matching its rendered depth at visible object pixels to `D_0` at those pixels

**Camera poses:**
- If provided externally, scale the trajectory to depth-space by matching VO-derived inter-frame baselines to depth-derived 3D displacements of tracked background points
- If solved jointly (default), camera poses start from identity and are optimized against background Gaussians (see Section 6.4)

### 6.3 Per-Entity Gaussian Initialization

**Background:**
- Unproject non-person, non-object pixels from `D_0` using depth-space coordinates
- Initialize background Gaussians on resulting point cloud

**Body (per person):**
- Initialize one Gaussian per body model vertex in canonical T-pose
- Inherit skinning weights from body mesh
- Set `smpl_vertex_id` for each Gaussian

**Hands (per hand, optional):**
- Initialize one Gaussian per MANO mesh vertex in canonical flat-hand pose
- Upsample in fingertip regions (higher density where detail matters)
- Inherit skinning weights from MANO mesh
- Set `mano_vertex_id` for each Gaussian

**Objects (per detected object):**
- Segment object with SAM2 in frame 0
- Run SAM 3D Objects on the cropped object image to obtain full 3D mesh (including hallucinated backside)
- Scale-align to depth-space (Section 6.2)
- Initialize Gaussians on SAM 3D mesh surface (sample ~5K Gaussians per object)
- Inherit normals from mesh for initial Gaussian orientations
- Inherit texture from mesh for initial SH coefficients

### 6.4 Camera Pose Modes

Camera poses are **optional input**. Three operating modes:

**Mode A — Static camera (default for fixed/slow-moving shots):**
Camera extrinsics fixed to identity throughout. Zero overhead, correct for table-top or tripod setups.

**Mode B — Poses solved jointly (default for moving camera, no VO required):**
Camera extrinsics initialized to identity and included as learnable parameters. Pose gradients flow exclusively through background Gaussians (dynamic entities have their own deformation models and do not contribute). Requires background to have sufficient texture. A `L_camera_smooth` regularizer on the trajectory is applied to prevent floater artifacts from compensating for camera error.

**Mode C — Poses from external VO/SfM:**
Initialize from pre-computed camera poses (Transform3d JSON per frame) and optionally refine during Phase 2. Fastest convergence and most robust for large egomotion.

### 6.5 Optional: Semantic Feature Lifting (Post-Processing)

Open-vocabulary semantic features are not part of the core optimization. If needed for downstream tasks (e.g., "find the mug" queries), they can be added after reconstruction:

- Run DINOv2 or CLIP on video frames to obtain per-pixel feature maps
- Lift features onto the frozen, optimized Gaussians using Feature 3DGS or the Splat Feature Solver
- This is a separate, lightweight step that does not affect reconstruction quality

## 7. Joint Optimization

### 7.1 Learnable Parameters

```
Per-Gaussian (all entities):
    position, rotation, scale, opacity, color_sh

Per-Gaussian (body/hand):
    skinning_weights

Per-frame:
    Body pose theta_t, translation t_t             (per person)
    MANO pose phi_t, translation u_t               (per hand, if provided)
    Object SE(3): R_t, t_t                         (per object)
    Camera extrinsics                              (Mode B only)

Shared across frames:
    Body shape beta                                (per person)
    MANO shape psi                                 (per hand, if provided)
    Non-rigid correction MLP weights               (optional, Phase 3)
```

### 7.2 Loss Functions

**Photometric (primary):**
```
L_rgb  = |render(gaussians, t) - I_t|_1
L_ssim = 1 - SSIM(render(gaussians, t), I_t)
```

**Depth regularization (soft, not hard):**
```
L_depth = |render_depth(t) - monocular_depth(t)|_1   (weighted low, ~0.1)
```

Because all entities were initialized in depth-space (Section 6.1), this loss operates in the same coordinate system as the scene geometry — there is no scale/shift mismatch. This prevents Gaussians from drifting to degenerate configurations but does not rigidly constrain their positions. The photometric loss remains the final arbiter where depth estimates are inaccurate.

**Entity mask supervision:**
```
L_mask     = BCE(render_entity_silhouette(t), SAM2_masks(t))
```

**Structural regularizers:**
```
L_skinning  = ||w||_1                           # sparse skinning weights (few bones per vertex)
L_rigid     = penalize non-rigid motion in object Gaussians
L_smooth    = ||param_t - param_{t-1}||^2       # temporal smoothness on poses
L_shape     = ||beta - beta_init||^2            # soft prior on body shape
L_camera_smooth = ||cam_t - 2*cam_{t-1} + cam_{t-2}||^2  # camera jerk (Mode B only)
```

**Loss weight hierarchy:**
```
L_rgb + L_ssim  : weight 1.0     # pixels are ground truth
L_mask          : weight 0.5     # SAM2 masks are reliable
L_depth         : weight 0.1     # useful prior, not gospel
L_smooth        : weight 0.01    # soft regularizer
L_skinning      : weight 0.01    # soft regularizer
L_rigid         : weight 0.01    # soft regularizer
L_shape         : weight 0.001   # very soft prior
L_camera_smooth : weight 0.01    # Mode B only
```

**No explicit contact loss.** Contact plausibility emerges from:
- Shared photometric optimization (hand and object Gaussians must explain the same pixels)
- Semantic mask consistency (hand and object boundaries must align with SAM2 predictions)
- Temporal smoothness (the hand can't teleport through the object)

### 7.3 Optimization Schedule

**Phase 1 — Global alignment (frames 0-10, ~100 iterations):**
- Optimize body shape `beta`, camera intrinsics, entity-to-depth-space scale factors
- Freeze individual Gaussian attributes
- Goal: establish correct coordinate alignment and rough pose

**Phase 2 — Joint reconstruction (all frames, ~5000 iterations):**
- Optimize all parameters jointly
- Enable Gaussian densification/pruning (entity-aware, see Section 8)
- Goal: high-fidelity reconstruction

**Phase 3 — Refinement (~2000 iterations):**
- Reduce learning rate
- Enable non-rigid correction MLP
- Goal: sharp details, correct clothing wrinkles, fine hand geometry

## 8. Entity-Aware Gaussian Density Control

Standard 3DGS adaptive density control (clone, split, prune), modified per entity:

**Hands:** Clone/split aggressively. Hands are small in image space but geometrically complex. Target ~10K-20K Gaussians per hand after densification.

**Body:** Moderate density. Clone near clothing boundaries and wrinkles. Target ~50K-100K Gaussians per person.

**Objects:** Clone near edges and texture boundaries. Split large Gaussians covering curved surfaces. Target ~5K-30K per object depending on size.

**Background:** Sparse, large Gaussians. Walls/floors need few Gaussians. Target ~100K-200K total.

**Inheritance rule:** New Gaussians created by clone/split inherit `entity_id`, `skinning_weights`, and `rigid_body_id` from their parent. This prevents entity label bleeding across boundaries.

## 9. Output Extraction

The optimized scene directly provides:

**Parametric model fits:**
- Body parameters `(theta_t, beta)` per person per frame — in `NlfResult`-compatible NPZ format
- MANO parameters `(phi_t, psi)` per hand per frame — same format (if hands were included)

**Gaussians:**
- Full scene as `.ply` (standard 3DGS format) with per-Gaussian `entity_id` stored as a scalar property
- Entity metadata in companion `entities.json` (entity_id ranges, rigid body IDs, body model type)

**Meshes:**
- Body mesh: evaluate body model at optimized parameters
- Object mesh: extract from object Gaussians via TSDF fusion or Poisson reconstruction (`.obj` with texture)
- Scene mesh: extract from background Gaussians

**Contact maps:**
- Identify hand Gaussians whose nearest object Gaussian is within threshold epsilon
- Or render hand and object Gaussians jointly and find pixels where both contribute

**Novel view rendering:**
- Standard 3DGS rasterization at any camera pose / time
- Render with or without specific entities (e.g., remove person, show only hands + objects)

## 10. Feasibility: Hardware Requirements

**Target hardware:** Single NVIDIA RTX 4090 (24 GB VRAM)

Estimated VRAM breakdown:
```
Background Gaussians (~200K)           :    ~12 MB
Body Gaussians (~50K per person)       :     ~4 MB
Hand Gaussians (~20K x 2)             :     ~3 MB
Object Gaussians (~30K)                :     ~2 MB
Body model parameters                  :     ~1 MB
Non-rigid correction MLP               :     ~5 MB
Rasterizer buffers (1080p)             :   ~500 MB
Optimizer states (Adam, 2x params)     :   ~200 MB
PyTorch overhead + fragmentation       :    ~2 GB
─────────────────────────────────────────────────
Total                                  :   ~3 GB
```

~21 GB headroom for larger scenes, higher resolution, or multiple people.

**Estimated training time (per-scene optimization):**
- Phase 1: ~2 minutes
- Phase 2: ~30-60 minutes
- Phase 3: ~15-30 minutes
- **Total: ~1-2 hours per video sequence on a single 4090**

**Rendering:** Real-time (>60 FPS) at 1080p after optimization, via standard 3DGS rasterization.

## 11. Software Stack

```
Framework:         PyTorch
Gaussian encoding: tiny-cuda-nn (fused hash grids + MLPs)
Rasterization:     diff-gaussian-rasterization (original 3DGS CUDA rasterizer)
Ray marching:      nerfacc (if hybrid volume rendering needed)
Body model:        smplx (PyTorch SMPL / SMPL+H / SMPL-X implementation)
Depth:             Any monocular depth module (MoGe, UniDepth, Depth Anything, ...)
Segmentation:      SAM2 (video instance segmentation + tracking)
3D reconstruction: SAM 3D Objects (single-image object mesh)
Body estimation:   v2d_nlf (SMPL / SMPL+H) — SMPL-X estimator optional future module
Hand estimation:   HaMeR (MANO) — optional future module
Camera poses:      Solved jointly (default) or from external VO/SfM
```

## 12. Key Assumptions and Limitations

**Assumptions:**
- Objects are rigid bodies (no deformable objects like cloth or rope)
- Camera is moving or objects are moving (need parallax for 3D)
- Video is reasonable quality (not extreme motion blur or very low resolution)

**Known limitations:**
- Topology changes (picking up / releasing objects) require the contact state to change between connected and disconnected Gaussians; not handled by the current formulation without special-casing
- Transparent / reflective objects violate the Gaussian opacity model
- Very fast hand motion may cause motion blur that degrades MANO fitting
- SAM 3D may hallucinate incorrect backside geometry for unusual objects; the optimization can correct this but convergence may be slower
- The coordinate system is in depth-model units, not necessarily metric. Metric scale requires either a metric depth model or a post-hoc calibration using known object dimensions (e.g., body height). Scenes without any known-size reference remain scale-ambiguous.
- Mode B camera optimization can produce floater Gaussians if background texture is insufficient for reliable pose gradients; fall back to Mode A (static camera) in such cases.

## 13. End-to-End Example

Using the `v2d_sam2` test assets (a video of a person interacting with an object, with bounding-box prompts for both).

### Prerequisite outputs

```python
import os
from v2d.common.utils import extract_images

VIDEO   = "modules/v2d_sam2/assets/test_video.mp4"
PROMPTS = "modules/v2d_sam2/assets/test_prompts.json"
WEIGHTS = "data/weights"
OUT     = "data/outputs/gsplat_example"

frames_dir   = f"{OUT}/frames"
depth_dir    = f"{OUT}/depth"
intrin_dir   = f"{OUT}/intrinsics"
masks_dir    = f"{OUT}/masks"      # SAM2 writes: masks_dir/{object_id}/{frame_id:06d}.png
objects_dir  = f"{OUT}/objects"
smpl_raw     = f"{OUT}/smpl/smpl_raw.npz"
smpl_aligned = f"{OUT}/smpl/smpl_aligned.npz"

# 1. Extract frames (needed for SAM3D single-image input)
extract_images(VIDEO, frames_dir)

# 2. Depth + intrinsics (MoGe)
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
run_video_to_depth(
    video_path=VIDEO,
    depth_folder=depth_dir,
    intrinsics_folder=intrin_dir,
    weights_path=f"{WEIGHTS}/moge",
)
# Reference intrinsics: use frame 0 (all frames share the same intrinsics from MoGe)
intrinsics_path = f"{intrin_dir}/000000.json"

# 3. Segmentation masks (SAM2)
#    test_prompts.json: object_id=0 → human, object_id=1 → object
#    Output layout: masks_dir/0/{frame:06d}.png, masks_dir/1/{frame:06d}.png
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
run_video_to_masks(
    video_path=VIDEO,
    prompts_path=PROMPTS,
    masks_dir=masks_dir,
    weights_dir=f"{WEIGHTS}/sam2",
)

# 4. Object mesh from frame 0 (SAM3D)
#    Uses the object mask (object_id=1) at frame 0
os.makedirs(objects_dir, exist_ok=True)
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
run_image_to_mesh(
    image_path=f"{frames_dir}/000000.png",
    mask_path=f"{masks_dir}/1/000000.png",
    mesh_path=f"{objects_dir}/object_1.obj",
    transform_path=f"{objects_dir}/object_1_transform.json",
    intrinsics_path=f"{objects_dir}/object_1_intrinsics.json",
    weights_dir=f"{WEIGHTS}/sam3d",
    with_texture_baking=True,
)

# 5. SMPL body estimation (NLF)
#    NLF expects masks at masks_dir/{frame:06d}.png — pass the human subdirectory
from v2d.nlf.docker.run_video_to_smpl import run_video_to_smpl
run_video_to_smpl(
    video_path=VIDEO,
    masks_dir=f"{masks_dir}/0",    # human masks only (object_id=0)
    intrinsics_path=intrinsics_path,
    gender="neutral",
    output_path=smpl_raw,
    weights_dir=f"{WEIGHTS}/nlf",
    model_type="smpl",
)

# 6. Align SMPL to depth-space
from v2d.nlf.docker.run_align_nlf_to_depth import run_align_nlf_to_depth
run_align_nlf_to_depth(
    smpl_results_path=smpl_raw,
    depth_folder=depth_dir,
    masks_dir=f"{masks_dir}/0",
    intrinsics_path=intrinsics_path,
    output_path=smpl_aligned,
    weights_dir=f"{WEIGHTS}/nlf",
)
```

### Gaussian splatting optimization

```python
# 7. Gsplat optimization
#    Reads entity roles from the SAM2 prompts JSON (role="human" → body entity,
#    role="object" + mesh provided → object entity).
from v2d.gsplat.docker.run_video_to_gsplat import run_video_to_gsplat

run_video_to_gsplat(
    video_path=VIDEO,
    depth_folder=depth_dir,
    intrinsics_path=intrinsics_path,
    masks_dir=masks_dir,                          # contains 0/ and 1/ subdirs
    prompts_path=PROMPTS,                         # provides role per object_id
    smpl_path=smpl_aligned,                       # optional: depth-aligned SMPL NPZ
    object_meshes={1: f"{objects_dir}/object_1.obj"},  # optional: SAM3D mesh per object_id
    output_dir=f"{OUT}/gsplat",
    weights_dir=f"{WEIGHTS}/gsplat",
    camera_mode="static",                         # or "joint" for moving camera
)

# Outputs:
#   data/outputs/gsplat_example/gsplat/gaussians.ply      — full scene
#   data/outputs/gsplat_example/gsplat/entities.json      — entity metadata
#   data/outputs/gsplat_example/gsplat/smpl/              — refined body params per frame
#   data/outputs/gsplat_example/gsplat/meshes/            — extracted object OBJs
#   data/outputs/gsplat_example/gsplat/renders/           — rendered video
```

### Data flow summary

```
test_video.mp4 ──┬─► MoGe ──────────────────────► depth/*.png + intrinsics/*.json
                 │                                           │
                 ├─► SAM2 ──────────────────────► masks/0/* (human)
                 │          test_prompts.json              masks/1/* (object)
                 │                                    │         │
                 │                    frames/000000.png         │
                 │                         │                    │
                 │                    SAM3D ◄───────────────────┘
                 │                         └──────────────────► objects/object_1.obj
                 │
                 └─► NLF (masks/0/) ─► smpl_raw.npz
                                             │
                              align_nlf_to_depth ─► smpl_aligned.npz
                                                          │
                              ┌───────────────────────────┤
                              │   depth/ + masks/ + objects/ + smpl_aligned.npz
                              ▼
                           gsplat ──────────────────────► gaussians.ply + renders/
```

## 14. Implementation Plan

### Module structure

```
v2d_gsplat/
├── lib/
│   ├── pyproject.toml             # deps: torch, diff-gaussian-rasterization,
│   │                              #       tiny-cuda-nn, smplx, trimesh, nerfacc
│   ├── run_video_to_gsplat.py     # lib entry point (CLI + importable)
│   ├── scene.py                   # GaussianScene: per-Gaussian tensors, entity slices
│   ├── deformation.py             # LBSDeformer (SMPL/SMPL+H/SMPL-X), Se3Deformer, StaticDeformer
│   ├── initialization.py          # init_background(), init_body(), init_object()
│   ├── losses.py                  # photometric, depth, mask, all regularizers
│   ├── densification.py           # entity-aware clone / split / prune
│   ├── optimization.py            # phase-scheduled training loop
│   └── extraction.py              # mesh via Poisson, contact maps, output serialization
├── docker/
│   ├── Dockerfile                 # pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel base
│   ├── build.py
│   ├── pyproject.toml             # no ML deps; pure orchestration
│   └── run_video_to_gsplat.py     # docker wrapper (CLI + importable)
└── assets/
```

### Phase 1 — Core Gaussian infrastructure (static scene)

Goal: render a static scene from a novel viewpoint.

- `scene.py`: `GaussianScene` dataclass holding per-Gaussian tensors (`positions`, `rotations`, `scales`, `opacities`, `sh_coeffs`, `entity_ids`). Supports slicing by entity_id. Standard 3DGS parameter layout compatible with `diff-gaussian-rasterization`.
- `initialization.py`: `init_background(depth_map, intrinsics, mask)` — unprojects non-entity pixels to a point cloud and seeds Gaussians there.
- `losses.py`: `loss_rgb` (L1) and `loss_ssim` against ground-truth frames.
- `optimization.py`: basic Adam loop with learning rate schedule, no densification yet.
- `rasterizer` call: thin wrapper around `diff-gaussian-rasterization` that takes a `GaussianScene` and camera parameters and returns an RGB image and depth map.
- Validation: overfit a single frame of the test video with background-only Gaussians.

### Phase 2 — Body integration (SMPL LBS)

Goal: body Gaussians track a person across frames via SMPL forward kinematics.

- `deformation.py`: `LBSDeformer` — loads SMPL/SMPL+H/SMPL-X model, exposes `deform(positions, skinning_weights, theta_t, beta)` that returns world-space positions for frame `t`. Autodiff-compatible.
- `initialization.py`: `init_body(smpl_result)` — places one Gaussian per SMPL vertex in canonical T-pose, copies skinning weights from SMPL mesh topology.
- `scene.py`: add `BodyAttributes` tensors (`smpl_vertex_id`, `skinning_weights`).
- `losses.py`: `loss_depth` (L1 against monocular depth) and `loss_mask` (BCE against SAM2 silhouette).
- `optimization.py`: Phase 1 and Phase 2 schedule (Phase 1 freezes Gaussians, fits `beta` and scale; Phase 2 unfreezes all).
- `densification.py`: entity-aware clone/split/prune — inherits `entity_id` and `skinning_weights` from parent Gaussian.
- Validation: body follows person in the test video; SMPL vertex positions match rendered silhouette.

### Phase 3 — Object integration (SE(3) rigid bodies)

Goal: object Gaussians track a rigid object across frames.

- `deformation.py`: `Se3Deformer` — per-frame `(R_t, t_t)` stored as learnable parameters; `deform(positions, rigid_body_id, t)` returns transformed positions.
- `initialization.py`: `init_object(mesh, depth_map, intrinsics, object_mask)` — samples ~5K Gaussians on the SAM3D mesh surface; scale-aligns mesh to depth-space by least-squares fit of rendered vs. observed depth at object pixels; initializes SH from mesh vertex colors.
- `losses.py`: `loss_rigid` — penalizes variance in relative Gaussian positions within an object over time.
- Object mesh output: `extraction.py` extracts OBJ with texture from object Gaussians via TSDF or Poisson.
- Validation: object Gaussians stay locked to the object in the test video.

### Phase 4 — Camera pose solving (Mode B)

Goal: camera extrinsics solved from background photometric consistency when no VO is provided.

- Camera extrinsics (`R_cam_t`, `t_cam_t`) added as learnable parameters (initialized to identity).
- Gradients enabled only for background Gaussians during camera update steps; dynamic entity Gaussians are detached from the camera pose graph.
- `L_camera_smooth` (jerk penalty on trajectory) added to `losses.py`.
- `optimization.py`: in Phase 1, alternate between Gaussian attribute steps and camera pose steps. In Phase 2, camera poses update jointly.
- Validation: reconstruct a short pan/tilt shot with no external VO input; compare recovered poses against ground-truth or COLMAP reference.

### Phase 5 — Refinement, output extraction, docker wrapper

Goal: full end-to-end pipeline matching the e2e example in Section 13.

- `optimization.py`: Phase 3 schedule — reduced LR, enable non-rigid correction MLP (`delta_MLP`; 3-layer MLP, 128 hidden, positional encoding on `x` concatenated with `pose_t`).
- `extraction.py`:
  - Body mesh: evaluate body model at final `(theta_t, beta)`.
  - Object mesh: TSDF fusion of per-frame depth + object mask → OBJ.
  - Contact map: nearest-Gaussian query between hand/body and object entity slices.
  - Serialize: `gaussians.ply` (with `entity_id` scalar property), `entities.json`, per-frame SMPL NPZ.
- `docker/run_video_to_gsplat.py`: full docker wrapper following repo conventions — `run_in_container()` with volume mounts for all inputs/outputs, `extra_args` for non-file params (`camera_mode`, optimization hyperparameters), dev mode support.
- End-to-end test with the `v2d_sam2` test assets (Section 13).

## 15. What Is Novel

Each individual component exists in published work:

| Component | Prior work |
|---|---|
| 4D Gaussian deformation fields | 4D-GS (CVPR 2024) |
| Per-Gaussian semantic features (post-processing) | Feature 3DGS (CVPR 2024) |
| Gaussians-on-SMPL with LBS | 3DGS-Avatar, HuGS |
| Hand-object 4D Gaussians | Interaction-Aware 4DGS (2025) |
| SMPL-X fitting via differentiable rendering | Multiple avatar papers |
| SAM 3D for object initialization | Meta SAM 3D (2025) |
| Video-consistent depth regularization | Video Depth Anything (CVPR 2025) |

**The contribution is the unified integration:** a single system that combines entity-aware Gaussian representations, structured parametric initialization (body model, SAM 3D), entity-specific deformation models, and joint photometric optimization — producing a complete, semantically decomposed 4D scene reconstruction from monocular video, with parametric model fits as a direct output.

No existing system reconstructs body + objects + background in a single differentiable framework with implicit contact handling and model-agnostic body parameterization.

## References

- Wu et al., "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering," CVPR 2024
- Duan et al., "4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives," ICLR 2025
- Zhou et al., "Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields," CVPR 2024
- Qian et al., "3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting," CVPR 2024
- Moreau et al., "Human Gaussian Splatting: Real-time Rendering of Animatable Avatars," CVPR 2024
- Yang et al., "Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction," 2025
- Meta AI, "SAM 3D Objects: 3Dfy Anything in Images," November 2025
- Chen et al., "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos," CVPR 2025
- Pavlakos et al., "Reconstructing Hands and Bodies with SMPL-X," CVPR 2019
- Romero et al., "Embodied Hands: Modeling and Capturing Hands and Bodies Together," SIGGRAPH Asia 2017
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," SIGGRAPH 2023
