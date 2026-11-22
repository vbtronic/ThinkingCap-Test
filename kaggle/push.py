#!/usr/bin/env python3
# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Push kaggle/kernel.py as a private GPU T4 x2 script kernel and start it.

    python kaggle/push.py smoke                         # quick check of every part
    python kaggle/push.py train --extra "--steps 200"
    python kaggle/push.py eval --checkpoint outputs/run1/ckpt-00200 --source <user>/<kernel>

Needs the Kaggle CLI and a token (~/.kaggle/access_token or KAGGLE_API_TOKEN).
The kernel clones the repo from GitHub at --commit (default: the local HEAD, which must be pushed).
Follow it with:  kaggle kernels status <id>   and   kaggle kernels output <id> -p <dir>
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("smoke", "train", "eval"))
    parser.add_argument("--user", default="viktorbrunclk")
    parser.add_argument("--slug", help="Kernel name (default: coldcap-<mode>)")
    parser.add_argument("--commit", help="Git commit to run (default: local HEAD)")
    parser.add_argument("--extra", default="", help="Extra flags for the training/eval script")
    parser.add_argument("--resume-from", default="", help="Path under /kaggle/input to resume from")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--source", action="append", default=[],
                        help="Kernel whose output to attach as input (user/slug), repeatable")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--kaggle", default="kaggle", help="How to call the CLI")
    arguments = parser.parse_args()

    commit = arguments.commit or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                text=True, check=True).stdout.strip()
    slug = arguments.slug or f"coldcap-{arguments.mode}"
    script = (HERE / "kernel.py").read_text()
    for name, value in (("MODE", arguments.mode), ("COMMIT", commit), ("EXTRA", arguments.extra),
                        ("RESUME_FROM", arguments.resume_from), ("CHECKPOINT", arguments.checkpoint)):
        script, count = re.subn(rf'^{name} = ".*?"', f"{name} = {json.dumps(value)}", script,
                                count=1, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"kernel.py has no {name} line")

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        (folder / f"{slug}.py").write_text(script)
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{arguments.user}/{slug}", "title": slug, "code_file": f"{slug}.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "model_sources": [],
            "kernel_sources": arguments.source, "machine_shape": arguments.accelerator,
        }, indent=2))
        command = arguments.kaggle.split() + ["kernels", "push", "-p", str(folder),
                                              "--accelerator", arguments.accelerator]
        print(f"pushing {arguments.user}/{slug} at {commit[:8]} ({arguments.mode})")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
