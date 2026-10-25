#!/usr/bin/env python3
r"""Build the README animation from bundled lossless frames.

Requires Python 3, ffmpeg and ffprobe. No Python packages are required.

Rebuild the README GIF and preview without external data:
    python scripts/make_demo_gifs.py --readme-only

Also rebuild the archived GIFs from the original videos:
    python scripts/make_demo_gifs.py \
        --area-video /path/to/M2S_RGB_to_Area_1080p_v3_2x.mp4 \
        --recognition-video /path/to/M2S_GT_vs_ACSH_1080p.mp4

Bundled frames were exported by the original renderer with only its main title
drawing call omitted. Predictions, values, panels and counters are unchanged.
Source videos are supplied separately. All inputs are checked by SHA-256;
the public manifest never records local paths.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ASSET_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets"
SOURCES = {
    "area": {
        "filename": "M2S_RGB_to_Area_1080p_v3_2x.mp4",
        "sha256": "c53fe5bf0f05d5dc2746b9f41402325295469fa82e6b904b04f1f8b5d4e5c96c",
        "duration_seconds": 24.0,
        "description": "Final v3: six independent held-out examples, four seconds each; all five panels visible simultaneously.",
    },
    "recognition": {
        "filename": "M2S_GT_vs_ACSH_1080p.mp4",
        "sha256": "7401c6301dfeceb87fca4d1259a9ed53e3c06fd45a35a807ea1393b3538f830c",
        "duration_seconds": 60.0,
        "description": "GT versus frozen ACSH predictions; two seconds of full field of view followed by four seconds of the same ROI per example.",
    },
    "area_titleless_frames": {
        "description": "Lossless native v3 compositions rerendered from the original cached inputs with only the main title drawing call omitted; no inference or estimator changes.",
        "original_renderer": "render_area_demo_v3.py",
        "original_renderer_sha256": "447e5bfe5faf74bd8825402cedef32da05f8751742968b8b46fd3ce98e0ef35a",
        "titleless_renderer": "render_area_demo_v3_titleless_20261025.py",
        "titleless_renderer_sha256": "844116af83cf10937840efbdeb2d4f81bcda0ba0939ee996edb97a9d33796d4e",
        "width": 1920,
        "height": 1080,
        "duration_per_frame_seconds": 4,
        "pixel_validation": "Each native frame differs from its original only in 6,819 main-title glyph pixels, within x=[41,584), y=[38,72). All other pixels match exactly. The native header background is uniformly RGB(243,245,247).",
        "frames": [
            {"file": "demo-frames/01.png", "case_id": "01", "sha256": "827b0d68fb2e9e1c794e1995903310761fe3bafb2c0255b6c021cd0dc6fc125e", "original_frame_sha256": "3b67736ad8866f79931b36f41dd6ec2fb6fde356bddc3188059aee5bf256fe6d"},
            {"file": "demo-frames/02.png", "case_id": "02", "sha256": "e0a0101c08a983d15067cba256e36378d70f2faf4286d8b885b8305880ed315d", "original_frame_sha256": "cee6bd445cbf20dc0fc16a159e7f9fbbd82690d0d34f8571f177f5a59d3b2e6a"},
        ],
    },
}

AREA_CASES = [
    ("01", "Overcast/camera_4073port_a_cam_1-1688437270.095890452", 0.35663455453363113, 0.3587742341733941),
    ("02", "Cloudy/camera_8554port_a_cam_1-1688520710.821112854", 0.26866948831482773, 0.23262974907305478),
    ("03", "Cloudy/camera_2726port_a_cam_1-1688532419.915283912", 0.1767443847562874, 0.20065378143170612),
    ("04", "Overcast/camera_598port_a_cam_1-1688436922.589267656", 0.21973275130300607, 0.2582104903225393),
    ("05", "Cloudy/camera_16533port_a_cam_1-1688521508.716086313", 0.1537650606232006, 0.16253534853878515),
    ("06", "Cloudy/camera_14119port_a_cam_1-1688521267.314475556", 0.20440682304766256, 0.21797912251798301),
]

CLIPS = [
    {"file": "area-pipeline.gif", "source": "area_titleless_frames", "start_seconds": 0, "duration_seconds": 8, "width": 960, "fps": 10, "case_ids": ["01", "02"], "caption": "RGB, instance segmentation, predicted depth, fitted road plane and calibrated box area; native examples 01–02 rendered without a main title."},
    {"file": "area-geometry.gif", "source": "area", "start_seconds": 8, "duration_seconds": 8, "width": 800, "fps": 10, "case_ids": ["03", "04"], "caption": "The same five-panel area pipeline on examples 03–04."},
    {"file": "area-multi-object.gif", "source": "area", "start_seconds": 16, "duration_seconds": 8, "width": 800, "fps": 10, "case_ids": ["05", "06"], "caption": "Multi-object scenes; the final area refers to the selected matched target, examples 05–06."},
    {"file": "segmentation-comparison.gif", "source": "recognition", "start_seconds": 0, "duration_seconds": 6, "width": 800, "fps": 10, "sample": "Overcast/camera_4804port_a_cam_1-1688437343.197702907", "caption": "Manual GT versus frozen ACSH prediction; full field of view followed by the same enlarged ROI."},
]


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path):
    return json.loads(run("ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate", "-of", "json", str(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readme-only", action="store_true", help="Use bundled native frames to rebuild only the README GIF and preview; retain archived assets.")
    parser.add_argument("--area-video", type=Path)
    parser.add_argument("--recognition-video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    args = parser.parse_args()
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            parser.error(f"{executable} is required")
    paths = {"area": args.area_video, "recognition": args.recognition_video}
    for frame in SOURCES["area_titleless_frames"]["frames"]:
        path = ASSET_DIR / frame["file"]
        if not path.is_file() or sha256(path) != frame["sha256"]:
            parser.error(f"Missing or modified bundled native frame: {path}")
    for key, path in ([] if args.readme_only else paths.items()):
        if path is None or not path.is_file():
            parser.error(f"Missing {key} source video: {path}")
        if sha256(path) != SOURCES[key]["sha256"]:
            parser.error(f"The {key} video does not match the original demonstration SHA-256. This extraction timeline is specific to that original.")
        if abs(float(probe(path)["format"]["duration"]) - SOURCES[key]["duration_seconds"]) > 0.05:
            parser.error(f"Unexpected {key} source duration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    manifest_path = args.output_dir / "media_manifest.json"
    if args.readme_only and manifest_path.is_file():
        assets = [asset for asset in json.loads(manifest_path.read_text())["assets"]
                  if asset["file"] not in ("area-pipeline.gif", "area-pipeline-preview.png")]
    for clip in (CLIPS[:1] if args.readme_only else CLIPS):
        output = args.output_dir / clip["file"]
        native = clip["source"] == "area_titleless_frames"
        input_args = ("-framerate", "1/4", "-start_number", "1", "-i", str(ASSET_DIR / "demo-frames" / "%02d.png")) if native else (
            "-ss", str(clip["start_seconds"]), "-t", str(clip["duration_seconds"]), "-i", str(paths[clip["source"]]))
        palette = "max_colors=256:reserve_transparent=0:stats_mode=full" if native else "max_colors=192:stats_mode=diff"
        quantization = "dither=none:diff_mode=rectangle" if native else "dither=bayer:bayer_scale=3:diff_mode=rectangle"
        filters = (
            f"[0:v]fps={clip['fps']},scale={clip['width']}:-1:flags=lanczos,split[a][b];"
            f"[a]palettegen={palette}[p];"
            f"[b][p]paletteuse={quantization}[out]"
        )
        run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "4",
            *input_args, "-filter_complex_threads", "1", "-filter_complex", filters,
            "-map", "[out]", "-t", str(clip["duration_seconds"]), "-an", "-loop", "0", str(output))
        details = probe(output)
        duration = float(details["format"]["duration"])
        if abs(duration - clip["duration_seconds"]) > 0.05:
            raise RuntimeError(f"Unexpected GIF duration: {output.name}: {duration}")
        if output.stat().st_size >= 8 * 1024 * 1024:
            raise RuntimeError(f"GIF exceeds the 8 MiB README size budget: {output.name}")
        asset = dict(clip)
        asset.update({"height": details["streams"][0]["height"], "bytes": output.stat().st_size, "sha256": sha256(output), "loop": "infinite"})
        assets.append(asset)
        print(f"{output.name}: {asset['width']}x{asset['height']}, {duration:g} s, {asset['bytes']:,} bytes")

    poster = args.output_dir / "area-pipeline-preview.png"
    run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(ASSET_DIR / "demo-frames" / "01.png"),
        "-frames:v", "1", "-vf", "scale=960:-1:flags=lanczos", "-threads", "1", str(poster))
    assets.append({"file": poster.name, "source": "area_titleless_frames", "case_id": "01", "width": 960, "height": 540, "bytes": poster.stat().st_size, "sha256": sha256(poster), "caption": "Native v3 example 01, rendered without a main title and downscaled for a static preview."})
    order = {clip["file"]: index for index, clip in enumerate(CLIPS)}
    assets.sort(key=lambda asset: order.get(asset["file"], len(CLIPS)))

    manifest = {
        "schema_version": 2,
        "generator": "scripts/make_demo_gifs.py",
        "ffmpeg_version": run("ffmpeg", "-version").splitlines()[0],
        "sources": SOURCES,
        "provenance": {
            "dataset": "M2S-RoAD, held-out test split",
            "model": "Frozen paper ACSH (P2 + ACmix) instance segmenter",
            "weights_sha256": "e5a809756eb4cdc0b86ee45343925a40cac2dcf7ead5311d23219ec5d7735f2a",
            "confidence_threshold": 0.25,
            "predictions": "All predictions at or above the uniform display threshold are retained in the source overview; no GT-based prediction correction or suppression.",
            "selection": "Curated qualitative examples; the GIFs are not a representative accuracy or runtime benchmark.",
            "temporal_scope": "Sparse independent annotated images with fixed presentation holds; no continuous driving footage, tracking, interpolated source frames or inference-FPS claim.",
            "area_definition": "Bounding-box planimetric footprint from fisheye rectification and local road-plane intersection, followed by frozen training-only log calibration; not mask area, cavity surface area or volume.",
            "reference": "LiDAR-derived evaluation reference areas, not surveyed ground truth; used only for comparison.",
            "plane_panel": "Native rectified RGB projected onto the actual fitted road plane, predicted mask contour, unmodified SVD support points and projected-box outline; appearance projection, not cavity reconstruction.",
            "depth_panel": "Frozen monocular predicted metric depth, not GT depth; per-example finite-positive P2–P98 display scale, fixed within each example.",
            "transforms": "The README GIF and preview are exported directly from bundled lossless native frames. The original renderer was rerun with only the main-title drawing call omitted, retaining its original uniform background. Each example is held for four seconds, downscaled with Lanczos and palette-quantized for 10 fps GIF playback. All other native pixels, predictions, numbers, panels and counters are unchanged. Archived GIFs retain their original video extraction, frame sampling, scaling and quantization.",
        },
        "area_cases": [
            {"case_id": case_id, "sample": sample, "source_start_seconds": i * 4, "source_duration_seconds": 4, "reference_area_m2": reference, "estimated_area_m2": estimate}
            for i, (case_id, sample, reference, estimate) in enumerate(AREA_CASES)
        ],
        "assets": assets,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
