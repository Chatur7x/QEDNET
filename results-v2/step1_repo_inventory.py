from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
print("BRANCH", subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip())
print("DATA_FILES")
for path in sorted((root / "data" / "raw").glob("*")):
    if path.is_file():
        print(path.relative_to(root).as_posix())
print("MODEL_CLASSES")
for rel in (
    "qednet/models/classical/logistic_regression.py",
    "qednet/models/classical/random_forest.py",
    "qednet/models/classical/rbf_svm.py",
    "qednet/models/classical/mlp.py",
    "qednet/models/classical/parameter_matched_mlp.py",
    "qednet/models/quantum/qsvm.py",
    "qednet/models/quantum/vqc.py",
    "qednet/models/quantum/reupload.py",
    "qednet/models/quantum/hybrid_qnn.py",
):
    path = root / rel
    print(f"{rel}: exists={path.exists()}")
print("PRIOR_ARTIFACTS_READ_ONLY: results/ contains prior files; no values from them are used as new benchmark results.")
