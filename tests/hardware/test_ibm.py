"""IBM hardware executor tests (dry-run mode, no credentials needed)."""

import numpy as np
from qiskit import QuantumCircuit

from qednet.hardware import HardwareResult, IBMCredentials, create_executor


def test_credentials_from_env(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    creds = IBMCredentials.from_env()
    assert not creds.is_configured()

    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-token")
    creds = IBMCredentials.from_env()
    assert creds.is_configured()
    assert creds.token == "test-token"


def test_create_executor_dry_run():
    executor = create_executor(dry_run=True)
    assert executor.dry_run is True
    assert executor.shots == 1024


def test_list_backends_dry_run():
    executor = create_executor(dry_run=True)
    backends = executor.list_backends()
    assert len(backends) == 1
    assert backends[0].name == "local_simulator"
    assert backends[0].simulator is True


def test_execute_dry_run_simple_circuit():
    """Test execution with a simple Qiskit circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    executor = create_executor(dry_run=True, shots=100)
    result = executor.execute(qc)

    assert isinstance(result, HardwareResult)
    assert result.success is True
    assert result.backend == "local_simulator"
    assert result.shots == 100
    assert result.runtime_s > 0
    assert result.transpiled_depth >= 0
    assert "counts" in result.__dict__
    assert sum(result.counts.values()) == 100


def test_transpile_circuit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    executor = create_executor(dry_run=True)
    transpiled, info = executor.transpile_circuit(qc)

    assert "depth" in info
    assert "gate_counts" in info
    assert "transpile_time_s" in info
    assert info["num_qubits"] == 2


def test_execute_with_params():
    """Test execution with parameterized circuit."""
    from qiskit.circuit import Parameter

    theta = Parameter("θ")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure_all()

    executor = create_executor(dry_run=True, shots=50)
    result = executor.execute(qc, params=np.array([0.5]))

    assert result.success is True
    assert sum(result.counts.values()) == 50
