"""D4 encoding tests: valid/invalid inputs, dims, execution, reproducibility."""

import numpy as np
import pytest

from qednet.encoding import (
    amplitude_encoding,
    angle_encoding,
    encode,
    get_encoding,
    statevector,
    validate_vector,
    zz_feature_map,
)


def test_valid_angle_runs():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def c(x):
        angle_encoding(x, wires=[0, 1])
        return qml.expval(qml.Z(0))

    assert np.isfinite(c(np.array([0.1, 0.2])))


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        validate_vector([0.1, 0.2, 0.3], n_qubits=2, encoding="angle")
    with pytest.raises(ValueError):
        validate_vector([0.1] * 9, n_qubits=3, encoding="amplitude")


def test_invalid_inputs():
    with pytest.raises(ValueError):
        validate_vector([], n_qubits=2, encoding="angle")
    with pytest.raises(ValueError):
        validate_vector([np.nan, 0.1], n_qubits=2, encoding="angle")
    with pytest.raises(ValueError):
        get_encoding("nope")


def test_zz_and_amplitude_execute():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def c1(x):
        zz_feature_map(x, wires=[0, 1])
        return qml.expval(qml.Z(0))

    @qml.qnode(dev)
    def c2(x):
        amplitude_encoding(x, wires=[0, 1])
        return qml.expval(qml.Z(0))

    assert np.isfinite(c1(np.array([0.3, -0.2])))
    assert np.isfinite(c2(np.array([0.3, -0.2, 0.5])))


def test_qubit_usage_and_reproducibility():
    s1 = statevector([0.5, -0.1], 2, "angle")
    s2 = statevector([0.5, -0.1], 2, "angle")
    assert s1.shape == (4,)
    np.testing.assert_allclose(s1, s2)
    # different inputs -> different states
    s3 = statevector([0.9, 0.1], 2, "angle")
    assert not np.allclose(s1, s3)


def test_dispatcher():
    assert get_encoding("zz") is zz_feature_map
    assert get_encoding("IQP") is zz_feature_map
    import pennylane as qml

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def c(x):
        encode("angle", x, wires=[0, 1])
        return qml.expval(qml.Z(0))

    assert np.isfinite(c(np.array([0.0, 0.0])))
