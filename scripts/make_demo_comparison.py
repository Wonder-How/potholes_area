"""Build the static five-stage comparison from native demonstration PNGs.

The input directory contains the original lossless case frames (01.png, etc.).
These research-renderer inputs are supplied separately from this repository.
Default columns are original cases 01, 02, 03 and 05, in their original order.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


ROWS = [
    {"stage": "Original + GT", "crop": [36, 112, 600, 472], "height": 472},
    {"stage": "Segmentation + box", "crop": [660, 112, 600, 472], "height": 472},
    {"stage": "Predicted depth", "crop": [1284, 112, 600, 472], "height": 472},
    {"stage": "Road-plane fit", "crop": [36, 604, 900, 420], "height": 280},
    {"stage": "Final area", "crop": [960, 604, 924, 420], "height": 273},
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory of original lossless case PNGs")
    parser.add_argument("--cases", nargs="+", default=["01", "02", "03", "05"])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "docs/assets/area-comparison.png")
    args = parser.parse_args()
    if len(args.cases) < 2 or len(set(args.cases)) != len(args.cases) or any(case not in {f"{i:02d}" for i in range(1, 7)} for case in args.cases):
        parser.error("Select at least two unique original case IDs from 01 through 06")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, gap, margin = 600, 12, 16
    inner_height = sum(row["height"] for row in ROWS) + gap * 4
    columns = len(args.cases)
    canvas_width = width * columns + gap * (columns-1) + margin * 2
    canvas_height = inner_height + margin * 2
    inputs, graph, sources = [], [], []
    for col, case in enumerate(args.cases):
        source = args.source / f"{case}.png"
        inputs.extend(["-i", str(source)])
        sources.append({"column": col+1, "case_id": source.stem, "source_sha256": sha(source)})
        graph.append(f"[{col}:v]split=5" + "".join(f"[c{col}r{row}]" for row in range(5)))
        for row, spec in enumerate(ROWS):
            x, y, w, h = spec["crop"]
            chain = f"[c{col}r{row}]crop={w}:{h}:{x}:{y}"
            if (w, h) != (width, spec["height"]):
                chain += f",scale={width}:{spec['height']}:flags=lanczos"
            chain += ",setsar=1"
            if row < 4:
                chain += f",pad={width}:{spec['height']+gap}:0:0:color=0xf3f5f7"
            graph.append(chain + f"[p{col}r{row}]")
        chain = "".join(f"[p{col}r{row}]" for row in range(5)) + "vstack=inputs=5"
        if col < columns-1:
            chain += f",pad={width+gap}:{inner_height}:0:0:color=0xf3f5f7"
        graph.append(chain + f"[column{col}]")
    graph.append("".join(f"[column{col}]" for col in range(columns))
                 + f"hstack=inputs={columns},pad={canvas_width}:{canvas_height}:{margin}:{margin}:color=0xf3f5f7,format=rgb24[out]")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
                    "-filter_complex_threads", "1", "-filter_complex", ";".join(graph),
                    "-map", "[out]", "-frames:v", "1", "-threads", "1", str(args.output)], check=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "layout": {"rows": 5, "columns": columns, "size": [canvas_width, canvas_height]},
        "rows": ROWS, "columns": sources,
        "transform": "Native cards cropped into a static grid. Last two rows resized proportionally. Original data, predictions, and displayed values retained.",
        "selection": "Curated examples chosen jointly for recognition quality, area agreement, clear geometry and scene diversity. Qualitative display, not a representative benchmark sample.",
        "original_case_order": args.cases,
        "generator": "scripts/make_demo_comparison.py",
        "output": args.output.name, "sha256": sha(args.output),
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
