import json

from runner import run_fold


def main():
    for repeat in range(1, 6):
        row = run_fold(
            "heart_disease",
            "qsvm",
            42,
            0,
            "A",
            "random",
            900,
            f"heart_qsvm_timing_repeat_{repeat}",
        )
        result = {
            "repeat": repeat,
            "status": row.get("status"),
            "fit_seconds": row.get("train_seconds"),
            "predict_seconds": row.get("predict_seconds"),
            "parent_elapsed_seconds": row.get("parent_elapsed_seconds"),
            "auc": row.get("auc"),
            "n_samples": row.get("n_samples"),
            "n_train": row.get("n_train"),
            "n_test": row.get("n_test"),
            "seed": row.get("seed"),
            "fold": row.get("fold"),
        }
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
