import json

from qednet.models import ParameterMatchedMLP


def main():
    candidates = {
        "no_hidden": (),
        "one_hidden_unit": (1,),
        "vqc_match_A": (1, 2),
        "reupload_match_A": (2, 1),
    }
    for dataset, input_dim in (("breast_cancer", 30), ("heart_disease", 13), ("parkinsons", 22), ("diabetes", 8), ("feature_setting_A", 4)):
        records = []
        for label, hidden_dims in candidates.items():
            model = ParameterMatchedMLP(n_params_target=16, input_dim=input_dim, hidden_dims=hidden_dims, random_state=42)
            model._build_model(input_dim)
            count = model.metadata()["n_params"]
            records.append({
                "architecture": label,
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "trainable_parameters": count,
                "within_10_percent_of_12": abs(count - 12) / 12 <= 0.10,
                "within_10_percent_of_16": abs(count - 16) / 16 <= 0.10,
            })
        print(json.dumps({"dataset": dataset, "candidates": records}, separators=(",", ":")))


if __name__ == "__main__":
    main()
