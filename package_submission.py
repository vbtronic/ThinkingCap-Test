#!/usr/bin/env python3
# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Build the ThinkingCap-Test submission zip (see TASK.md, "Submission").

    python package_submission.py --best outputs/run1/ckpt-00120 \\
        --others outputs/run1/ckpt-00040 outputs/run1/ckpt-00080 \\
        --results runs/eval/results.json --comparison runs/eval/comparison.json \\
        --summary COLLABORATION.md --hf https://huggingface.co/<user>/<repo>

Writes dist/<name>.zip with: the best LoRA at the root (adapter_model.safetensors,
adapter_config.json, metrics.json), other-checkpoints/<ckpt>/, <name>.bundle (git bundle
--all), results.json, comparison.json, IDEA.md, the collaboration summary and SUBMISSION.md
(links and contents). Refuses to go over the 500 MB the submission page accepts.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
LIMIT = 500 * 1024 * 1024
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "metrics.json")
GITHUB = "https://github.com/vbtronic/ThinkingCap-Test"


def copy_adapter(source, target):
    source = Path(source)
    if not (source / "adapter_model.safetensors").exists():
        raise SystemExit(f"{source} has no adapter_model.safetensors")
    target.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="viktor-brunclik")
    p.add_argument("--best", required=True, help="Checkpoint directory of the best LoRA")
    p.add_argument("--others", nargs="*", default=[], help="More checkpoints along the curve")
    p.add_argument("--results", required=True, help="results.json written by evaluate.py")
    p.add_argument("--comparison", help="comparison.json written by eval_pair.py / compare.py")
    p.add_argument("--summary", required=True, help="Collaboration summary (agent work)")
    p.add_argument("--hf", required=True, help="Hugging Face model repo with the LoRA")
    p.add_argument("--out", default="dist")
    arguments = p.parse_args()

    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise SystemExit("Commit everything first: the bundle must hold the code you ran")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()

    out = Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        stage = Path(folder) / arguments.name
        stage.mkdir()
        subprocess.run(["git", "bundle", "create", str(stage / f"{arguments.name}.bundle"), "--all"],
                       cwd=ROOT, check=True)
        copy_adapter(arguments.best, stage)
        others = []
        for checkpoint in arguments.others:
            name = Path(checkpoint).name
            copy_adapter(checkpoint, stage / "other-checkpoints" / name)
            others.append(name)
        shutil.copy2(arguments.results, stage / "results.json")
        if arguments.comparison:
            shutil.copy2(arguments.comparison, stage / "comparison.json")
        shutil.copy2(ROOT / "IDEA.md", stage / "IDEA.md")
        shutil.copy2(arguments.summary, stage / "COLLABORATION.md")

        results = json.loads(Path(arguments.results).read_text())
        lines = [
            "# ColdCap — ThinkingCap-Test submission",
            "",
            f"- Code: {GITHUB} (commit `{commit[:12]}`; the same history is in "
            f"`{arguments.name}.bundle`)",
            f"- LoRA on Hugging Face: {arguments.hf}",
            f"- Best checkpoint: `{Path(arguments.best).name}`, at the root of this zip "
            "(`adapter_model.safetensors`, `adapter_config.json`, `metrics.json`)",
            f"- Base model: {results.get('model')} @ {results.get('revision')}",
            f"- Evaluated on: {json.dumps(results.get('hardware'))}",
            "",
            "Contents:",
            "",
            "| File | What |",
            "|---|---|",
            "| `adapter_model.safetensors`, `adapter_config.json` | best LoRA (rank 16) |",
            "| `metrics.json` | its validation numbers during training |",
            "| `other-checkpoints/` | more points on the length / accuracy curve: "
            + (", ".join(others) or "none") + " |",
            "| `results.json` | `evaluate.py` output for the best LoRA |",
            "| `comparison.json` | the comparison with the base model scored on the same setup |",
            f"| `{arguments.name}.bundle` | `git bundle --all` of the code |",
            "| `IDEA.md` | what was tried and why, results, hardware |",
            "| `COLLABORATION.md` | how the work was split with the coding agent |",
            "",
            "Evaluate: `python evaluate.py --adapter <this folder> --out runs/coldcap`.",
        ]
        (stage / "SUBMISSION.md").write_text("\n".join(lines) + "\n")

        target = out / f"{arguments.name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage))
    size = target.stat().st_size
    if size > LIMIT:
        raise SystemExit(f"{target} is {size / 2**20:.0f} MB, over the 500 MB limit")
    print(f"wrote {target} ({size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
