import json

from runner import run_fold


def main():
    for dataset, model in (("heart_disease", "parameter_matched_mlp_reupload"), ("diabetes", "parameter_matched_mlp_vqc")):
        for seed in (42, 1729, 2024):
            for fold in (0, 1, 2):
                row = run_fold(dataset, model, seed, fold, "B", "random", 900, "benchmark")
                print(json.dumps({
                    "dataset": dataset,
                    "model": model,
                    "feature_setting": "B",
                    "seed": seed,
                    "fold": fold,
                    "status": row.get("status"),
                    "trainable_parameters": row.get("trainable_parameters"),
                    "parent_elapsed_seconds": row.get("parent_elapsed_seconds"),
                }), flush=True)


if __name__ == "__main__":
    main()
