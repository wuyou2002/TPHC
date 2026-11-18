import argparse
import json
from pathlib import Path

import datasets
import lm_eval
# lm_eval 0.4.x uses lazy imports: submodules must be imported explicitly
# before access (otherwise AttributeError: module 'lm_eval' has no attribute 'tasks')
import lm_eval.tasks
from lm_eval.tasks import TaskManager

import models


datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="outputs/tinyllama-tphc4")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tasks", type=str, nargs="+", default=["hellaswag", "openbookqa", "winogrande", "arc_challenge", "arc_easy", "piqa"])
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output", type=str, default="harness/tinyllama-tphc4.json")

    args = parser.parse_args()

    # Use local task overrides: this machine has no internet, so the six tasks'
    # dataset_path is redirected to local dataset dirs (harness/tasks/*.yaml).
    # include_path registers them alongside the built-in tasks (same names).
    task_manager = TaskManager(include_path=str(Path(__file__).parent / "harness" / "tasks"))
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model="hf",
        model_args=f"pretrained={args.model_name_or_path},dtype={args.dtype},attn_implementation=sdpa",
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        log_samples=False,
        task_manager=task_manager,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, default=lambda o: '<not serializable>')

if __name__ == "__main__":
    main()
