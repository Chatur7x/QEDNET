import numpy as np

from qednet.explain.explainer import _param_shift_grad
from qednet.models.quantum.vqc import VQC


def test_vqc_parameter_shift_gradient_uses_statevector_package_import():
    model = VQC(n_qubits=2, layers=1, random_state=42)
    model.params = {"W": np.array([[0.4, 0.8]]), "alpha": np.array(2.0)}
    features = np.array([[0.2, 0.7], [0.5, 0.1]])

    gradient = _param_shift_grad(model, features, layer=0, qubit=0)

    assert gradient.shape == (2,)
    assert np.isfinite(gradient).all()
