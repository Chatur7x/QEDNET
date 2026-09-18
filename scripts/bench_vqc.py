"""Benchmark VQC training speed to calibrate experiment design."""
import time
import numpy as np
import pennylane as qml
import autograd.numpy as anp
from autograd import grad

n_qubits, n_layers, n_train = 8, 3, 400
rng = np.random.default_rng(42)
X = rng.uniform(-np.pi, np.pi, size=(n_train, n_qubits)).astype(np.float64)
y = rng.integers(0, 2, size=n_train).astype(np.float64)

dev = qml.device("lightning.qubit", wires=n_qubits)

def circuit(features, weights):
    qml.AngleEmbedding(features=features, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

qnode = qml.QNode(circuit, dev, diff_method="adjoint")
weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
W = rng.uniform(0, 2*np.pi, size=weights_shape)

def loss(W):
    preds = anp.array([qnode(x, W) for x in X[:64]])
    return anp.mean((preds - (2*y[:64]-1))**2)

t0 = time.time()
for _ in range(3):
    l = loss(W)
t1 = time.time()
print(f"forward 64 samples: {(t1-t0)/3:.3f}s -> {(t1-t0)/3/64*1000:.2f} ms/sample")

g = grad(loss)
t0 = time.time()
for _ in range(2):
    gw = g(W)
t1 = time.time()
print(f"forward+backward 64 samples: {(t1-t0)/2:.3f}s")

# batch execution test
tapes = []
for x in X[:64]:
    with qml.queuing.AnnotatedQueue() as q:
        qml.AngleEmbedding(features=x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(W, wires=range(n_qubits))
        qml.expval(qml.PauliZ(0))
    tapes.append(qml.tape.QuantumScript.from_queue(q))
t0 = time.time()
res = qml.execute(tapes, dev, diff_method="adjoint")
t1 = time.time()
print(f"qml.execute 64 tapes: {t1-t0:.3f}s -> {(t1-t0)/64*1000:.2f} ms/tape")

# statevector kernel approach benchmark
dev2 = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev2)
def state_circuit(features):
    qml.AngleEmbedding(features=features, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(W, wires=range(n_qubits))
    return qml.state()
t0 = time.time()
states = np.array([state_circuit(x) for x in X[:100]])
t1 = time.time()
print(f"statevector 100 samples: {t1-t0:.3f}s -> {(t1-t0)/100*1000:.2f} ms/sample")
