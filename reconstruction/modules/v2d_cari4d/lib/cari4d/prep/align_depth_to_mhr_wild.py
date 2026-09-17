from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import Utils
from lib_mhr import MHRLayer, MHR_PARAM_DIMS, assert_mhr_schema, decode_mhr_vertices_numpy, load_mhr_result
from prep.align_utils import compute_scale_robust_log
from prep.mhr_depth_h5 import DepthFrameRecord, MHRDepthH5Writer
from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
from prep.mhr_export_utils import MHR_CAMERA_NAMES, camera_calibration, frame_names, load_edex, read_mask, transform_points
from prep.mhr_wild_depth import DEFAULT_WILD_DEPTH_BACKEND, load_wild_depth_report
from tools.pipeline_timing import PipelineTimer


WILD_DEPTH_ALIGNMENT_METHOD = "huber-log-ratio-to-direct-mhr-depth-v1"


def _identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _render_human_depth(vertices_camera: np.ndarray, K: np.ndarray, image_shape: tuple[int, int], glctx: object, mesh_tensors: dict[str, torch.Tensor]) -> np.ndarray:
    height, width = image_shape
    vertices = torch.as_tensor(vertices_camera, device="cuda", dtype=torch.float)
    depth, _ = Utils.nvdiff_color_depth_render(np.repeat(K[None], len(vertices), axis=0), glctx, mesh_tensors, (height, width), vertices, depth_only=True)
    return depth.detach().cpu().numpy().astype(np.float32)


def _raw_depth_iterator(path: Path):
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r")
        if values.dtype != np.uint16 or values.ndim != 3:
            raise ValueError(f"Raw depth NPY must be uint16 [T,H,W], got shape={values.shape}, dtype={values.dtype}")
        return iter(values)
    from videoio import Uint16Reader

    return iter(Uint16Reader(str(path)))


def align_monocular_depth_to_mhr_wild(export_seq: str | Path, raw_depth_video: str | Path, mhr_init: str | Path, output: str | Path, *, depth_report: str | Path, camera_id: int = 0, render_batch_size: int = 4, encoding_workers: int = 16, write_batch_size: int = 64, redo: bool = False, profiler: PipelineTimer | None = None) -> Path:
    setup_started = profiler.start() if profiler is not None else 0.0
    export_seq, raw_depth_video, mhr_init, output = map(lambda value: Path(value).resolve(), (export_seq, raw_depth_video, mhr_init, output))
    depth_backend = DEFAULT_WILD_DEPTH_BACKEND
    depth_report = Path(depth_report).resolve()
    if camera_id != 0:
        raise ValueError(f"The wild export currently contains only camera 0, got {camera_id}")
    for path in (raw_depth_video, mhr_init):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not depth_report.is_file():
        raise FileNotFoundError(depth_report)
    load_wild_depth_report(depth_report, expected_depth=raw_depth_video)
    if render_batch_size <= 0 or encoding_workers <= 0 or write_batch_size <= 0:
        raise ValueError("render_batch_size, encoding_workers, and write_batch_size must be positive")
    init = load_mhr_result(mhr_init)
    assert_mhr_schema(init, require_geometry=True)
    names = frame_names(export_seq, camera_id)
    if [str(value) for value in init["frames"]] != names or len(init["mhr_trans"]) != len(names):
        raise ValueError("Direct MHR initialization timeline differs from the wild RGB export")
    K, world_to_camera = camera_calibration(load_edex(export_seq), camera_id)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ.get("MHR_ASSETS_ROOT"), device="cuda")
    vertices_world = decode_mhr_vertices_numpy(layer, {key: np.asarray(init[key], dtype=np.float32) for key in MHR_PARAM_DIMS}, batch_size=render_batch_size)
    vertices_camera = transform_points(vertices_world.reshape(-1, 3), world_to_camera).reshape(vertices_world.shape)
    faces_raw = layer.mesh_faces(device=torch.device("cuda"))
    faces = faces_raw.detach().cpu().numpy() if torch.is_tensor(faces_raw) else np.asarray(faces_raw)
    faces = np.asarray(faces, dtype=np.int32)
    import nvdiffrast.torch as dr

    glctx = dr.RasterizeCudaContext()
    mesh_tensors = {"faces": torch.as_tensor(faces, device="cuda", dtype=torch.int)}
    raw_iterator = _raw_depth_iterator(raw_depth_video)
    first_raw = np.asarray(next(raw_iterator), dtype=np.uint16).astype(np.float32) / 1000.0
    image_shape = tuple(first_raw.shape)
    camera_name = MHR_CAMERA_NAMES[camera_id]
    input_identity = {"schema": WILD_DEPTH_ALIGNMENT_METHOD, "depth_backend": depth_backend, "depth_model_id": MOGE2_MODEL_ID, "depth_model_revision": MOGE2_MODEL_REVISION, "depth_source_commit": MOGE2_SOURCE_COMMIT, "export": _identity(export_seq / "wild_export.json"), "raw_depth": _identity(raw_depth_video), "raw_depth_report": _identity(depth_report), "direct_mhr": _identity(mhr_init)}
    output.parent.mkdir(parents=True, exist_ok=True)
    if profiler is not None:
        profiler.record("setup_and_mhr_decode", setup_started)
        profiler.update_metadata({"depth_backend": depth_backend, "frames": len(names), "render_batch_size": int(render_batch_size), "encoding_workers": int(encoding_workers), "write_batch_size": int(write_batch_size)})
    with MHRDepthH5Writer(output, {camera_name: names}, redo=redo, alignment_method=WILD_DEPTH_ALIGNMENT_METHOD, alignment_input_identity=input_identity, encoding_workers=encoding_workers) as writer:
        pending = []
        progress = tqdm(total=len(names), desc=f"Align {depth_backend} to direct MHR")
        for start in range(0, len(names), render_batch_size):
            stop = min(start + render_batch_size, len(names))
            decode_started = profiler.start() if profiler is not None else 0.0
            raw_chunk = [first_raw] if start == 0 else []
            raw_chunk.extend(np.asarray(next(raw_iterator), dtype=np.uint16).astype(np.float32) / 1000.0 for _ in range(stop - start - len(raw_chunk)))
            if any(tuple(value.shape) != image_shape for value in raw_chunk):
                raise ValueError(f"Raw {depth_backend} frame shape changed within [{start}, {stop})")
            if profiler is not None:
                profiler.record("raw_depth_decode", decode_started)
            render_started = profiler.start() if profiler is not None else 0.0
            rendered_chunk = _render_human_depth(vertices_camera[start:stop], K, image_shape, glctx, mesh_tensors)
            if profiler is not None:
                profiler.record("human_depth_render", render_started)
            alignment_started = profiler.start() if profiler is not None else 0.0
            for local_index, (raw_depth, human_depth) in enumerate(zip(raw_chunk, rendered_chunk)):
                index = start + local_index
                frame_name = names[index]
                if writer.has_frame(camera_name, index):
                    progress.update(1)
                    continue
                human_mask = read_mask(export_seq, "human", camera_id, frame_name)
                if human_mask.shape != raw_depth.shape:
                    raise ValueError(f"Wild human mask and depth shapes differ at {frame_name}: {human_mask.shape} != {raw_depth.shape}")
                valid = human_mask & np.isfinite(raw_depth) & np.isfinite(human_depth) & (raw_depth > 0.001) & (human_depth > 0.001)
                valid_count = int(valid.sum())
                if valid_count < 100:
                    raise ValueError(f"Wild human-anchored depth alignment has only {valid_count} corresponding pixels at {frame_name}")
                scale = compute_scale_robust_log(raw_depth, human_depth, valid)
                aligned = raw_depth.copy()
                aligned[aligned > 0.001] *= float(scale)
                pending.append(DepthFrameRecord(index, raw_depth, aligned, float(scale), 0.0, valid_count))
                if len(pending) >= write_batch_size:
                    writer.submit_frames(camera_name, pending)
                    pending.clear()
                progress.update(1)
            if profiler is not None:
                profiler.record("mask_scale_and_h5_write", alignment_started)
        finalize_started = profiler.start() if profiler is not None else 0.0
        if pending:
            writer.submit_frames(camera_name, pending)
        writer.flush_submitted()
        progress.close()
        try:
            next(raw_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError(f"Raw {depth_backend} timeline has more than {len(names)} frames")
        writer.mark_complete()
        if profiler is not None:
            profiler.record("final_write_and_commit", finalize_started)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Align pinned MoGe 2 wild-video depth to direct SAM 3D Body MHR depth at corresponding human pixels.")
    parser.add_argument("export_seq")
    parser.add_argument("--raw-depth", "--raw-depth-video", dest="raw_depth", required=True)
    parser.add_argument("--depth-report", required=True)
    parser.add_argument("--mhr-init", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--encoding-workers", type=int, default=16)
    parser.add_argument("--write-batch-size", type=int, default=64)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    with PipelineTimer("wild_depth_alignment") as profiler:
        print(align_monocular_depth_to_mhr_wild(args.export_seq, args.raw_depth, args.mhr_init, args.output, depth_report=args.depth_report, camera_id=args.camera_id, render_batch_size=args.render_batch_size, encoding_workers=args.encoding_workers, write_batch_size=args.write_batch_size, redo=args.redo, profiler=profiler))


if __name__ == "__main__":
    main()
