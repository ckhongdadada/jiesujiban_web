from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path("C:/Users/28414/anaconda3/envs/qwen_env/python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_unit_classifier_hybrid.py"
DATA_PATH = Path("C:/Users/28414/Desktop/\u7559\u8a00\u677f\u5408\u5e76\u6570\u636e_\u6700\u7ec8\u8bad\u7ec3\u7248_v3.xlsx")
SAVE_DIR = PROJECT_ROOT / "final_model_hybrid_v3_32cls"


def build_command(mode: str, extra_args: list[str]) -> list[str]:
    command = [
        str(PYTHON_EXE),
        str(TRAIN_SCRIPT),
        "--data-path",
        str(DATA_PATH),
        "--save-dir",
        str(SAVE_DIR),
        "--init-from",
        "",
    ]

    if mode == "fresh":
        command.extend(["--train-from-scratch", "--no-resume"])

    command.extend(extra_args)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely start hybrid v3 32-class training without overwriting final_model_hybrid."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fresh",
        dest="mode",
        action="store_const",
        const="fresh",
        default="fresh",
        help="Start a new training run from scratch. This is the default mode.",
    )
    mode_group.add_argument(
        "--resume",
        dest="mode",
        action="store_const",
        const="resume",
        help="Resume from final_model_hybrid_v3_32cls/training_state.pt.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow fresh mode to write to an output directory that already contains model artifacts.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to train_unit_classifier_hybrid.py, for example: --epochs 3",
    )
    args = parser.parse_args()

    if not PYTHON_EXE.exists():
        print(f"ERROR: qwen_env Python not found: {PYTHON_EXE}")
        return 1
    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: training script not found: {TRAIN_SCRIPT}")
        return 1
    if not DATA_PATH.exists():
        print(f"ERROR: training data not found: {DATA_PATH}")
        print("Expected desktop file: liuyanban-hebing-shuju_final-training_v3.xlsx Chinese filename")
        return 1

    existing_artifacts = [
        SAVE_DIR / "pytorch_model.bin",
        SAVE_DIR / "training_state.pt",
        SAVE_DIR / "label_map.json",
    ]
    if args.mode == "fresh" and not args.allow_existing_output and any(path.exists() for path in existing_artifacts):
        print(f"Refusing to overwrite existing training artifacts in: {SAVE_DIR}")
        print("For a new run in this directory, add: --allow-existing-output")
        print("For checkpoint continuation, use: --resume")
        return 2

    if args.mode == "resume" and not (SAVE_DIR / "training_state.pt").exists():
        print(f"ERROR: resume requested but checkpoint not found: {SAVE_DIR / 'training_state.pt'}")
        print("Use --fresh to start a new training run.")
        return 3

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    extra_args = args.training_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = build_command(args.mode, extra_args)
    print("Starting hybrid v3 32-class training")
    print(f"  data_path: {DATA_PATH}")
    print(f"  save_dir:  {SAVE_DIR}")
    if args.mode == "fresh":
        print("  mode: fresh training, no resume, no old 109-class init weights")
    else:
        print("  mode: resume from training_state.pt, no old 109-class init weights")
    print("  command:")
    print("  " + " ".join(f'"{item}"' if " " in item else item for item in command))
    print()

    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
