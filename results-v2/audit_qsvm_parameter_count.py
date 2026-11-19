import json

from qednet.models import QSVM

model = QSVM(n_qubits=4, encoding="zz", device="default.qubit", shots=None, random_state=42)
print(json.dumps({"class": type(model).__name__, "n_qubits": model.n_qubits, "encoding": model.encoding, "trainable_parameters": model.n_params}, sort_keys=True))
