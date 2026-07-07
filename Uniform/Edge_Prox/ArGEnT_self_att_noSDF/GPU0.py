import os
import sys
import time
import json
import argparse
import torch
from pathlib import Path



# Ensure directory is on path for import resolution
this_dir = Path(__file__).parent.resolve()
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

from Training_script import main as train_main  # noqa: E402


PRESETS_GPU0 = [
    "M",
    "S",
    "L",
]


def run_with_fallback(preset: str, initial_batch: int) -> bool:
    """Try training with a sequence of decreasing batch sizes upon OOM."""
    iterative = max(1, int(initial_batch * 0.1))
    batch_plan = list(range(initial_batch, 0, -iterative))
    batch_plan.append(1)
    # REMOVED: batch_plan = [1]  (was overriding initial_batch, forcing batch=1 always)
    for b in batch_plan:
        try:
            print(f"\n[GPU0] Preset={preset} | Trying batch={b}")
            train_main(preset, b)
            print(f"[GPU0] Preset={preset} | Completed with batch={b}")
            return True
        except RuntimeError as e:
            low = str(e).lower()
            if ("out of memory" in low or "cuda" in low) and b != 1:
                print(f"[GPU0] OOM/CUDA at batch {b}; reducing and retrying...")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2)
                continue
            print(f"[GPU0] Non-recoverable RuntimeError for preset {preset}: {e}")
            return False
        except Exception as e:
            print(f"[GPU0] Unexpected error for preset {preset}: {e}")
            return False
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more training presets with automatic batch fallback on OOM."
    )
    parser.add_argument(
        "--preset",
        nargs="+",
        default=None,
        help=(
            "Preset name(s) to run. Use one or many values, or comma-separated values. "
            "Use 'all' to run all presets in the JSON file."
        ),
    )
    parser.add_argument(
        "--initial-batch",
        type=int,
        default=200,
        help="Initial batch size to try before fallback reductions.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    return parser.parse_args()


def load_available_presets() -> list[str]:
    presets_path = this_dir / "model_presets.json"
    if not presets_path.exists():
        raise FileNotFoundError(f"Preset file not found: {presets_path}")

    with open(presets_path, "r", encoding="utf-8") as f:
        all_presets = json.load(f)

    if not isinstance(all_presets, dict):
        raise RuntimeError("model_presets.json must contain a JSON object at the top level")

    return sorted(all_presets.keys())


def resolve_requested_presets(raw_presets, available_presets: list[str]) -> list[str]:
    if not raw_presets:
        return PRESETS_GPU0

    parsed = []
    for token in raw_presets:
        for item in token.split(","):
            name = item.strip()
            if name:
                parsed.append(name)

    if len(parsed) == 1 and parsed[0].lower() == "all":
        return available_presets

    allowed = set(available_presets)
    unknown = [p for p in parsed if p not in allowed]
    if unknown:
        raise ValueError(
            "Unknown preset(s): "
            + ", ".join(unknown)
            + "\nAvailable presets: "
            + ", ".join(available_presets)
        )
    return parsed


def main() -> None:
    args = parse_args()
    available_presets = load_available_presets()

    if args.list_presets:
        print("Available presets:")
        for preset in available_presets:
            print(f"- {preset}")
        return

    if args.initial_batch < 1:
        raise ValueError("--initial-batch must be >= 1")

    print("Starting GPU0 preset run set...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(project_dir).parent  # parent folder name is model name
    model_name = parent_dir.name
    print(model_name)

    selected_presets = resolve_requested_presets(args.preset, available_presets)
    for preset in selected_presets:
        run_with_fallback(preset, initial_batch=args.initial_batch)
    print("GPU0 run set finished.")


if __name__ == "__main__":
    main()
