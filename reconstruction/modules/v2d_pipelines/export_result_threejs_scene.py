#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a V2D world result bundle to a small Three.js verification scene."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import types
from pathlib import Path
from urllib.request import urlopen
from typing import Any

import numpy as np
from PIL import Image
import trimesh


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>V2D World Verification</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; font-family: system-ui, sans-serif; background: #111; color: #f2f2f2; }
    #viewer { position: fixed; inset: 0; }
    #hud {
      position: fixed; left: 16px; top: 16px; width: min(520px, calc(100vw - 32px));
      background: rgba(16, 18, 22, 0.86); border: 1px solid rgba(255,255,255,0.18);
      border-radius: 6px; padding: 12px; backdrop-filter: blur(8px);
    }
    #hud h1 { margin: 0 0 8px; font-size: 15px; font-weight: 650; letter-spacing: 0; }
    #row { display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: center; }
    #frameSlider { width: 100%; }
    #toggles { display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 10px; font-size: 13px; }
    label { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
    button {
      color: #f2f2f2; background: #2a2f37; border: 1px solid rgba(255,255,255,0.2);
      border-radius: 5px; height: 28px; padding: 0 9px; cursor: pointer;
    }
    #stats { margin-top: 8px; color: #cbd0d6; font-size: 12px; line-height: 1.35; }
    #legend { position: fixed; right: 16px; bottom: 16px; color: #ddd; font-size: 12px; background: rgba(16,18,22,0.78); padding: 10px; border-radius: 6px; }
    .x { color: #ff5959; } .y { color: #54d66f; } .z { color: #6697ff; } .g { color: #ffb13d; }
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="hud">
    <h1>V2D world verification</h1>
    <div id="row">
      <button id="prevBtn">Prev</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0" />
      <button id="nextBtn">Next</button>
    </div>
    <div id="toggles">
      <label><input id="showDepth" type="checkbox" checked />Depth</label>
      <label><input id="showObject" type="checkbox" checked />Object</label>
      <label><input id="showHands" type="checkbox" checked />Hands</label>
      <label><input id="showCameras" type="checkbox" checked />Cameras</label>
      <label><input id="showGrid" type="checkbox" checked />Grid</label>
    </div>
    <div id="stats"></div>
  </div>
  <div id="legend">
    <div><span class="x">red</span> +X/right</div>
    <div><span class="y">green</span> +Y/forward</div>
    <div><span class="z">blue</span> +Z/up</div>
    <div><span class="g">orange</span> gravity/down</div>
  </div>
  <script>
    const statsEl = () => document.getElementById("stats");
    window.addEventListener("error", (event) => {
      const el = statsEl();
      if (el) el.textContent = "viewer error: " + (event.message || event.error || "unknown error");
    });
    window.addEventListener("unhandledrejection", (event) => {
      const el = statsEl();
      const reason = event.reason && event.reason.message ? event.reason.message : event.reason;
      if (el) el.textContent = "viewer error: " + reason;
    });
  </script>
  <script src="./scene_data.js"></script>
  <script type="module">
    import * as THREE from "./three.module.js";
    import { OrbitControls } from "./OrbitControls.js";

    document.getElementById("stats").textContent = "loading Three.js scene...";
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const data = window.V2D_SCENE;
    const container = document.getElementById("viewer");
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x111318, 1);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.01, 1000);
    camera.up.set(0, 0, 1);
    camera.position.set(data.view.eye[0], data.view.eye[1], data.view.eye[2]);
    camera.lookAt(data.view.target[0], data.view.target[1], data.view.target[2]);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(data.view.target[0], data.view.target[1], data.view.target[2]);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(2.5, -3.5, 5.0);
    scene.add(sun);

    function mat4(rowMajor) {
      const m = new THREE.Matrix4();
      m.set(
        rowMajor[0][0], rowMajor[0][1], rowMajor[0][2], rowMajor[0][3],
        rowMajor[1][0], rowMajor[1][1], rowMajor[1][2], rowMajor[1][3],
        rowMajor[2][0], rowMajor[2][1], rowMajor[2][2], rowMajor[2][3],
        rowMajor[3][0], rowMajor[3][1], rowMajor[3][2], rowMajor[3][3],
      );
      return m;
    }

    function setMatrix(obj, rowMajor) {
      obj.matrix.copy(mat4(rowMajor));
      obj.matrix.decompose(obj.position, obj.quaternion, obj.scale);
      obj.updateMatrixWorld(true);
    }

    function makeLine(points, color, opacity = 1, width = 1) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(points.flat(), 3));
      const mat = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, linewidth: width });
      return new THREE.LineSegments(geom, mat);
    }

    function makeGrid(size, divisions) {
      const pts = [];
      const half = size * 0.5;
      for (let i = 0; i <= divisions; i++) {
        const t = -half + size * (i / divisions);
        pts.push([-half, t, 0], [half, t, 0], [t, -half, 0], [t, half, 0]);
      }
      return makeLine(pts, 0x3b424c, 0.55);
    }

    function makeArrow(start, end, color) {
      const s = new THREE.Vector3(...start);
      const e = new THREE.Vector3(...end);
      const dir = new THREE.Vector3().subVectors(e, s);
      const len = dir.length();
      return new THREE.ArrowHelper(dir.normalize(), s, len, color, Math.min(len * 0.18, 0.08), Math.min(len * 0.08, 0.035));
    }

    const worldGroup = new THREE.Group();
    worldGroup.add(makeArrow([0,0,0], [data.axis_length,0,0], 0xff4040));
    worldGroup.add(makeArrow([0,0,0], [0,data.axis_length,0], 0x48d66a));
    worldGroup.add(makeArrow([0,0,0], [0,0,data.axis_length], 0x5f8fff));
    worldGroup.add(makeArrow([0,0,0], [0,0,-data.axis_length], 0xffaa2c));
    scene.add(worldGroup);

    const grid = makeGrid(data.grid.size, data.grid.divisions);
    scene.add(grid);

    const meshGeom = new THREE.BufferGeometry();
    meshGeom.setAttribute("position", new THREE.Float32BufferAttribute(data.object_mesh.vertices.flat(), 3));
    meshGeom.setIndex(data.object_mesh.faces.flat());
    meshGeom.computeVertexNormals();
    const objectMesh = new THREE.Mesh(
      meshGeom,
      new THREE.MeshStandardMaterial({ color: 0xc49b68, roughness: 0.78, metalness: 0.04, side: THREE.DoubleSide })
    );
    objectMesh.matrixAutoUpdate = false;
    const objectAxes = new THREE.AxesHelper(data.axis_length * 0.7);
    objectMesh.add(objectAxes);
    scene.add(objectMesh);

    const depthGeom = new THREE.BufferGeometry();
    const depthMat = new THREE.PointsMaterial({ size: data.depth_point_size, vertexColors: true, transparent: true, opacity: 0.78 });
    const depthPoints = new THREE.Points(depthGeom, depthMat);
    scene.add(depthPoints);

    function frustumLocal(intr, depth) {
      const w = intr.width, h = intr.height, fx = intr.fx, fy = intr.fy, cx = intr.cx, cy = intr.cy;
      const corners = [[0,0], [w,0], [w,h], [0,h]].map(([u,v]) => [(u-cx)/fx*depth, (v-cy)/fy*depth, depth]);
      return [
        [0,0,0], corners[0], [0,0,0], corners[1], [0,0,0], corners[2], [0,0,0], corners[3],
        corners[0], corners[1], corners[1], corners[2], corners[2], corners[3], corners[3], corners[0],
      ];
    }

    const cameraGhostGroup = new THREE.Group();
    const activeCameraGroup = new THREE.Group();
    scene.add(cameraGhostGroup);
    scene.add(activeCameraGroup);
    for (const f of data.frames) {
      const group = new THREE.Group();
      const line = makeLine(frustumLocal(f.intrinsics, data.frustum_depth), 0x8b949e, 0.28);
      group.add(line);
      group.add(new THREE.AxesHelper(data.axis_length * 0.35));
      setMatrix(group, f.camera_to_world);
      cameraGhostGroup.add(group);
    }

    const handGroups = { left: new THREE.Group(), right: new THREE.Group() };
    scene.add(handGroups.left);
    scene.add(handGroups.right);

    const handMeshes = {};
    for (const side of ["left", "right"]) {
      if (data.hand_meshes && data.hand_meshes[side] && data.hand_meshes[side].faces && data.hand_meshes[side].frames.length) {
        const geom = new THREE.BufferGeometry();
        geom.setIndex(data.hand_meshes[side].faces.flat());
        geom.setAttribute("position", new THREE.Float32BufferAttribute([], 3));
        const mat = new THREE.MeshStandardMaterial({
          color: side === "left" ? 0xb06ae6 : 0x3bb7c6,
          roughness: 0.55,
          transparent: true,
          opacity: 0.68,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geom, mat);
        handGroups[side].add(mesh);
        handMeshes[side] = mesh;
      }
    }

    function setFrame(i) {
      const frame = data.frames[i];
      document.getElementById("frameSlider").value = i;
      setMatrix(objectMesh, frame.object_to_world);

      depthGeom.setAttribute("position", new THREE.Float32BufferAttribute(frame.depth.points.flat(), 3));
      depthGeom.setAttribute("color", new THREE.Float32BufferAttribute(frame.depth.colors.flat(), 3));
      depthGeom.computeBoundingSphere();

      activeCameraGroup.clear();
      const active = new THREE.Group();
      active.add(makeLine(frustumLocal(frame.intrinsics, data.frustum_depth * 1.25), 0xf0f4ff, 0.85));
      active.add(new THREE.AxesHelper(data.axis_length * 0.55));
      setMatrix(active, frame.camera_to_world);
      activeCameraGroup.add(active);

      for (const side of ["left", "right"]) {
        handGroups[side].clear();
        const h = frame.hands[side];
        if (h && h.valid) {
          const axes = new THREE.AxesHelper(data.axis_length * 0.45);
          const marker = new THREE.Mesh(
            new THREE.SphereGeometry(data.axis_length * 0.035, 16, 8),
            new THREE.MeshStandardMaterial({ color: side === "left" ? 0xb06ae6 : 0x3bb7c6 })
          );
          const wristGroup = new THREE.Group();
          wristGroup.add(axes);
          wristGroup.add(marker);
          setMatrix(wristGroup, h.wrist_to_world);
          handGroups[side].add(wristGroup);
        }
        if (handMeshes[side] && data.hand_meshes[side].frames[i]) {
          const mesh = handMeshes[side];
          mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.hand_meshes[side].frames[i].vertices.flat(), 3));
          mesh.geometry.computeVertexNormals();
          mesh.geometry.computeBoundingSphere();
          handGroups[side].add(mesh);
        }
      }

      const handMode = data.meta.hand_mesh_status;
      document.getElementById("stats").textContent =
        `frame ${frame.frame_index} | depth points ${frame.depth.points.length} | object faces ${data.object_mesh.faces.length} | hands ${handMode}`;
    }

    function applyVisibility() {
      depthPoints.visible = document.getElementById("showDepth").checked;
      objectMesh.visible = document.getElementById("showObject").checked;
      handGroups.left.visible = document.getElementById("showHands").checked;
      handGroups.right.visible = document.getElementById("showHands").checked;
      cameraGhostGroup.visible = document.getElementById("showCameras").checked;
      activeCameraGroup.visible = document.getElementById("showCameras").checked;
      grid.visible = document.getElementById("showGrid").checked;
    }

    const slider = document.getElementById("frameSlider");
    slider.max = String(data.frames.length - 1);
    slider.addEventListener("input", () => setFrame(Number(slider.value)));
    document.getElementById("prevBtn").addEventListener("click", () => setFrame(Math.max(0, Number(slider.value) - 1)));
    document.getElementById("nextBtn").addEventListener("click", () => setFrame(Math.min(data.frames.length - 1, Number(slider.value) + 1)));
    for (const id of ["showDepth", "showObject", "showHands", "showCameras", "showGrid"]) {
      document.getElementById(id).addEventListener("change", applyVisibility);
    }

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    setFrame(0);
    applyVisibility();
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""



def _rodrigues(r: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(r, dtype=np.float64) / theta
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _install_chumpy_stub() -> None:
    if "chumpy" in sys.modules:
        return

    class _Ch:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, x=None, *args, **kwargs):
            self._state = {"x": np.asarray(x)} if x is not None else {}

        def __setstate__(self, state):
            self._state = state if isinstance(state, dict) else {"x": state}

        def _resolve(self) -> np.ndarray:
            state = self._state
            if "x" in state:
                x = state["x"]
                return x._resolve() if isinstance(x, _Ch) else np.asarray(x)
            if "a" in state and "idxs" in state:
                a = state["a"]
                a = a._resolve() if isinstance(a, _Ch) else np.asarray(a)
                result = a.reshape(-1)[np.asarray(state["idxs"])]
                preferred_shape = state.get("preferred_shape")
                return result.reshape(preferred_shape) if preferred_shape is not None else result
            return np.array([])

        def __array__(self, dtype=None):
            return np.asarray(self._resolve(), dtype=dtype)

        @property
        def r(self):
            return self._resolve()

    class _ChumpyModule(types.ModuleType):
        def __getattr__(self, name: str):
            return _Ch

    stub = _ChumpyModule("chumpy")
    stub.__path__ = []
    stub.Ch = _Ch
    sys.modules["chumpy"] = stub
    for submodule in ("reordering", "utils", "ch", "logic"):
        child = _ChumpyModule(f"chumpy.{submodule}")
        sys.modules[f"chumpy.{submodule}"] = child
        setattr(stub, submodule, child)


def _resolve_mano_right_path(root: Path) -> Path:
    candidates = (
        root / "MANO_RIGHT.pkl",
        root / "models" / "MANO_RIGHT.pkl",
        root / "mano" / "MANO_RIGHT.pkl",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find MANO_RIGHT.pkl under {root}")


def _load_mano_right_model(root: Path) -> dict[str, np.ndarray]:
    _install_chumpy_stub()
    path = _resolve_mano_right_path(root)
    with path.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    model = {
        "v_template": np.asarray(raw["v_template"], dtype=np.float64),
        "shapedirs": np.asarray(raw["shapedirs"], dtype=np.float64),
        "posedirs": np.asarray(raw["posedirs"], dtype=np.float64),
        "weights": np.asarray(raw["weights"], dtype=np.float64),
        "faces": np.asarray(raw["f"], dtype=np.int32),
        "parents": np.asarray(raw["kintree_table"][0], dtype=np.int32),
    }
    model["J_regressor"] = np.asarray(raw["J_regressor"].todense(), dtype=np.float64)
    return model


def _mano_forward_camera(
    model: dict[str, np.ndarray],
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    is_right: bool,
    hand_scale: float,
) -> np.ndarray:
    full_pose = np.concatenate(
        [np.asarray(global_orient, dtype=np.float64), np.asarray(hand_pose, dtype=np.float64)],
        axis=0,
    )
    v_shaped = model["v_template"] + np.einsum("ijk,k->ij", model["shapedirs"], np.asarray(betas, dtype=np.float64))
    J = model["J_regressor"] @ v_shaped
    R = np.stack([_rodrigues(full_pose[3 * i : 3 * i + 3]) for i in range(16)], axis=0)
    pose_feature = (R[1:] - np.eye(3, dtype=np.float64)).reshape(-1)
    v_posed = v_shaped + np.einsum("ijk,k->ij", model["posedirs"], pose_feature)

    parents = model["parents"]
    G = np.zeros((16, 4, 4), dtype=np.float64)
    for joint_idx in range(16):
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = R[joint_idx]
        local[:3, 3] = J[joint_idx] if joint_idx == 0 else J[joint_idx] - J[parents[joint_idx]]
        G[joint_idx] = G[parents[joint_idx]] @ local if joint_idx > 0 else local

    G_final = np.zeros((16, 4, 4), dtype=np.float64)
    for joint_idx in range(16):
        offset = np.eye(4, dtype=np.float64)
        offset[:3, 3] = -J[joint_idx]
        G_final[joint_idx] = G[joint_idx] @ offset

    T = np.einsum("vk,kij->vij", model["weights"], G_final)
    v_homo = np.concatenate([v_posed, np.ones((len(v_posed), 1), dtype=np.float64)], axis=1)
    verts = np.einsum("vij,vj->vi", T, v_homo)[:, :3]
    if not is_right:
        verts[:, 0] *= -1.0
    if abs(float(hand_scale) - 1.0) > 1e-8:
        center = verts.mean(axis=0, keepdims=True)
        verts = (verts - center) * float(hand_scale) + center
    return verts + np.asarray(transl, dtype=np.float64)[None, :]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _round_array(a: np.ndarray, digits: int = 5) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        if fallback is None:
            raise ValueError("Cannot normalize near-zero vector")
        return _normalize(fallback)
    return v / n


def _frame_indices(n_frames: int, frames: str | None, num_frames: int, manifest: dict) -> list[int]:
    if frames:
        out = []
        for item in frames.split(","):
            item = item.strip()
            if not item:
                continue
            idx = int(item)
            if idx < 0:
                idx += n_frames
            if not 0 <= idx < n_frames:
                raise ValueError(f"Frame index {idx} is outside [0, {n_frames})")
            out.append(idx)
        return sorted(dict.fromkeys(out))

    gravity = manifest.get("gravity_alignment") or {}
    sample_frames = gravity.get("sample_frame_indices") or []
    sample_frames = [int(i) for i in sample_frames if 0 <= int(i) < n_frames]
    if sample_frames:
        return sorted(dict.fromkeys(sample_frames))
    return sorted(dict.fromkeys(np.linspace(0, n_frames - 1, num_frames).round().astype(int).tolist()))


def _default_sibling_dir(result_dir: Path, name: str) -> Path:
    return result_dir.parent / name


def _load_intrinsics(path: Path, fallback: np.ndarray | None = None) -> dict[str, float]:
    if path.exists():
        data = _load_json(path)
        return {
            "fx": float(data["fx"]),
            "fy": float(data.get("fy", data["fx"])),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
        }
    if fallback is None:
        raise FileNotFoundError(path)
    return {
        "fx": float(fallback[0]),
        "fy": float(fallback[1]),
        "cx": float(fallback[2]),
        "cy": float(fallback[3]),
        "width": 1280,
        "height": 800,
    }


def _load_metric_depth(depth_path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth = np.full(raw.shape, np.inf, dtype=np.float32)
    valid = raw > 0
    depth[valid] = 1.0 / (raw[valid] / 65535.0) - 1.0
    return depth


def _sample_depth_world(
    frame_idx: int,
    c2w: np.ndarray,
    intr: dict[str, float],
    depth_dir: Path | None,
    frames_dir: Path | None,
    stride: int,
    max_points: int,
    max_depth: float,
) -> dict[str, list]:
    if depth_dir is None:
        return {"points": [], "colors": []}
    depth_path = depth_dir / f"{frame_idx:06d}.png"
    if not depth_path.exists():
        return {"points": [], "colors": []}

    depth = _load_metric_depth(depth_path)
    h, w = depth.shape
    ys = np.arange(stride // 2, h, stride, dtype=np.int32)
    xs = np.arange(stride // 2, w, stride, dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth)
    uu = uu[valid]
    vv = vv[valid]
    z = z[valid]

    if len(z) > max_points:
        pick = np.linspace(0, len(z) - 1, max_points).round().astype(np.int64)
        uu, vv, z = uu[pick], vv[pick], z[pick]

    x = (uu.astype(np.float64) - intr["cx"]) / intr["fx"] * z
    y = (vv.astype(np.float64) - intr["cy"]) / intr["fy"] * z
    pts_cam = np.stack([x, y, z], axis=1)
    pts_world = pts_cam @ c2w[:3, :3].T + c2w[:3, 3][None, :]

    colors = np.full((len(pts_world), 3), 0.62, dtype=np.float64)
    if frames_dir is not None:
        frame_path = frames_dir / f"{frame_idx:06d}.png"
        if frame_path.exists():
            rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float64) / 255.0
            if rgb.shape[0] == h and rgb.shape[1] == w:
                colors = rgb[vv, uu]

    return {
        "points": _round_array(pts_world, 5),
        "colors": _round_array(colors, 4),
    }


def _load_mesh(mesh_path: Path, max_faces: int) -> dict[str, list]:
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices)]
        if not meshes:
            raise ValueError(f"No mesh geometry in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max_faces > 0 and len(faces) > max_faces:
        step = int(math.ceil(len(faces) / max_faces))
        faces = faces[::step][:max_faces]
    return {
        "vertices": _round_array(vertices, 6),
        "faces": faces.astype(int).tolist(),
    }


def _compute_view(arrays: dict[str, np.ndarray], frame_ids: list[int]) -> dict[str, list]:
    pts = [np.zeros((1, 3), dtype=np.float64)]
    for idx in frame_ids:
        for key in ("camera_to_world_transform", "object_to_world_transform", "hand_left_wrist_to_world_transform", "hand_right_wrist_to_world_transform"):
            if key in arrays:
                pts.append(np.asarray(arrays[key][idx, :3, 3], dtype=np.float64)[None, :])
    pts_np = np.concatenate(pts, axis=0)
    lo = pts_np.min(axis=0)
    hi = pts_np.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 0.5)
    eye = center + np.array([radius * 1.8, -radius * 2.3, radius * 1.4], dtype=np.float64)
    return {"target": _round_array(center, 5), "eye": _round_array(eye, 5), "radius": radius}


def _try_hand_meshes(
    arrays: dict[str, np.ndarray],
    frame_ids: list[int],
    mano_assets_root: Path | None,
) -> tuple[dict[str, Any], str]:
    empty = {"left": {"faces": [], "frames": []}, "right": {"faces": [], "frames": []}}
    if mano_assets_root is None:
        return empty, "wrist axes only: no --mano_assets_root"

    try:
        model = _load_mano_right_model(mano_assets_root)
    except Exception as exc:
        return empty, f"wrist axes only: MANO model load failed ({exc})"

    out: dict[str, Any] = {}
    n_frames_total = int(arrays["camera_to_world_transform"].shape[0])
    default_scales = np.ones((n_frames_total,), dtype=np.float64)

    for side in ("left", "right"):
        prefix = f"hand_{side}"
        valid_key = f"{prefix}_is_valid"
        if valid_key not in arrays or not bool(np.asarray(arrays[valid_key])[frame_ids].any()):
            out[side] = {"faces": [], "frames": []}
            continue

        faces = np.asarray(model["faces"], dtype=np.int32)
        if side == "left":
            # Vertex x is mirrored for left hands; reverse winding so normals stay outward.
            faces = faces[:, [0, 2, 1]]

        betas = np.asarray(arrays[f"{prefix}_betas"], dtype=np.float64)
        hand_scales = np.asarray(arrays.get(f"{prefix}_scale", default_scales), dtype=np.float64)
        frames = []
        for frame_idx in frame_ids:
            verts_cam = _mano_forward_camera(
                model=model,
                global_orient=np.asarray(arrays[f"{prefix}_wrist_orient_in_camera"][frame_idx], dtype=np.float64),
                hand_pose=np.asarray(arrays[f"{prefix}_finger_pose"][frame_idx], dtype=np.float64).reshape(45),
                betas=betas,
                transl=np.asarray(arrays[f"{prefix}_wrist_trans_in_camera"][frame_idx], dtype=np.float64),
                is_right=(side == "right"),
                hand_scale=float(hand_scales[frame_idx]),
            )
            c2w = np.asarray(arrays["camera_to_world_transform"][frame_idx], dtype=np.float64)
            verts_world = verts_cam @ c2w[:3, :3].T + c2w[:3, 3][None, :]
            frames.append({"frame_index": int(frame_idx), "vertices": _round_array(verts_world, 5)})
        out[side] = {"faces": faces.astype(int).tolist(), "frames": frames}

    exported = [side for side in ("left", "right") if out.get(side, {}).get("frames")]
    if not exported:
        return empty, "wrist axes only: no valid hand tracks"
    return out, "MANO meshes (NumPy LBS): " + ", ".join(exported)


def _write_three_vendor_assets(output_dir: Path) -> None:
    """Write local Three.js modules so the viewer does not depend on import maps.

    OrbitControls from the CDN imports bare ``three``. We patch that import to
    the local ``./three.module.js`` file, which avoids browser import-map
    compatibility and stale CDN/module-cache issues.
    """
    assets = {
        "three.module.js": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
        "OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js",
    }
    failures = []
    for name, url in assets.items():
        dst = output_dir / name
        if dst.exists() and dst.stat().st_size > 1024:
            continue
        try:
            text = urlopen(url, timeout=10).read().decode("utf-8")
        except Exception as exc:
            failures.append(f"{name} from {url}: {exc}")
            continue
        if name == "OrbitControls.js":
            text = text.replace("from 'three';", "from './three.module.js';")
            text = text.replace('from "three";', 'from "./three.module.js";')
        dst.write_text(text)

    missing = [name for name in assets if not (output_dir / name).exists() or (output_dir / name).stat().st_size <= 1024]
    if missing:
        details = "; ".join(failures) if failures else "unknown download failure"
        raise RuntimeError(
            "Three.js viewer assets are missing: "
            + ", ".join(missing)
            + ". Re-run with network access, or copy three.module.js and OrbitControls.js "
            + "from another threejs_scene directory. "
            + details
        )


def export_scene(
    result_dir: Path,
    output_dir: Path,
    frames_dir: Path | None,
    depth_dir: Path | None,
    intrinsics_dir: Path | None,
    frames: str | None,
    num_frames: int,
    depth_stride: int,
    max_depth_points: int,
    max_depth: float,
    max_object_faces: int,
    mano_assets_root: Path | None,
) -> dict[str, Any]:
    arrays = dict(np.load(result_dir / "result.npz", allow_pickle=False))
    manifest = _load_json(result_dir / "manifest.json") if (result_dir / "manifest.json").exists() else {}
    n_frames = int(arrays["camera_to_world_transform"].shape[0])
    frame_ids = _frame_indices(n_frames, frames, num_frames, manifest)

    if frames_dir is None:
        src = (manifest.get("sources") or {}).get("frames_dir")
        frames_dir = Path(src) if src else None
    if depth_dir is None:
        candidate = _default_sibling_dir(result_dir, "depth")
        depth_dir = candidate if candidate.exists() else None
    if intrinsics_dir is None:
        candidate = _default_sibling_dir(result_dir, "intrinsics")
        intrinsics_dir = candidate if candidate.exists() else None

    mesh_path = result_dir / "mesh.obj"
    if not mesh_path.exists():
        mesh_src = ((manifest.get("mesh") or {}).get("mesh")) or "mesh.obj"
        mesh_path = result_dir / mesh_src
    object_mesh = _load_mesh(mesh_path, max_object_faces)
    hand_meshes, hand_mesh_status = _try_hand_meshes(arrays, frame_ids, mano_assets_root)

    frame_entries = []
    for idx in frame_ids:
        intr_path = intrinsics_dir / f"{idx:06d}.json" if intrinsics_dir is not None else Path("__missing__")
        intr = _load_intrinsics(intr_path, arrays.get("camera_intrinsics"))
        c2w = np.asarray(arrays["camera_to_world_transform"][idx], dtype=np.float64)
        depth = _sample_depth_world(
            frame_idx=idx,
            c2w=c2w,
            intr=intr,
            depth_dir=depth_dir,
            frames_dir=frames_dir,
            stride=depth_stride,
            max_points=max_depth_points,
            max_depth=max_depth,
        )
        hands = {}
        for side in ("left", "right"):
            valid_key = f"hand_{side}_is_valid"
            T_key = f"hand_{side}_wrist_to_world_transform"
            valid = valid_key in arrays and bool(arrays[valid_key][idx])
            hands[side] = {
                "valid": valid,
                "wrist_to_world": _round_array(arrays[T_key][idx], 6) if valid and T_key in arrays else None,
            }
        frame_entries.append({
            "frame_index": int(idx),
            "camera_to_world": _round_array(c2w, 6),
            "object_to_world": _round_array(arrays["object_to_world_transform"][idx], 6),
            "intrinsics": intr,
            "hands": hands,
            "depth": depth,
        })

    view = _compute_view(arrays, frame_ids)
    radius = float(view["radius"])
    scene_data = {
        "meta": {
            "result_dir": str(result_dir.resolve()),
            "frame_indices": frame_ids,
            "world_coordinate_convention": manifest.get("world_coordinate_convention"),
            "gravity_alignment": manifest.get("gravity_alignment"),
            "hand_mesh_status": hand_mesh_status,
        },
        "axis_length": max(radius * 0.18, 0.08),
        "frustum_depth": max(radius * 0.16, 0.08),
        "depth_point_size": max(radius * 0.006, 0.004),
        "grid": {"size": max(radius * 3.2, 1.0), "divisions": 20},
        "view": {"target": view["target"], "eye": view["eye"]},
        "object_mesh": object_mesh,
        "hand_meshes": hand_meshes,
        "frames": frame_entries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_three_vendor_assets(output_dir)
    (output_dir / "index.html").write_text(HTML)
    with (output_dir / "scene_data.js").open("w") as f:
        f.write("window.V2D_SCENE = ")
        json.dump(scene_data, f, separators=(",", ":"))
        f.write(";\n")
    return scene_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--frames_dir", type=Path, default=None)
    parser.add_argument("--depth_dir", type=Path, default=None)
    parser.add_argument("--intrinsics_dir", type=Path, default=None)
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices. Defaults to gravity sample frames.")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--depth_stride", type=int, default=24)
    parser.add_argument("--max_depth_points", type=int, default=2500)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--max_object_faces", type=int, default=20000)
    parser.add_argument("--mano_assets_root", type=Path, default=None)
    args = parser.parse_args()

    data = export_scene(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        depth_dir=args.depth_dir,
        intrinsics_dir=args.intrinsics_dir,
        frames=args.frames,
        num_frames=args.num_frames,
        depth_stride=max(1, args.depth_stride),
        max_depth_points=max(1, args.max_depth_points),
        max_depth=args.max_depth,
        max_object_faces=args.max_object_faces,
        mano_assets_root=args.mano_assets_root,
    )
    print(f"Wrote Three.js scene: {args.output_dir / 'index.html'}")
    print(f"  frames: {data['meta']['frame_indices']}")
    print(f"  hand meshes: {data['meta']['hand_mesh_status']}")
    print(f"  depth points/frame: {[len(f['depth']['points']) for f in data['frames']]}")


if __name__ == "__main__":
    main()
