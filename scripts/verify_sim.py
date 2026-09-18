"""Quick sanity check of the batched simulator vs PennyLane."""
import sys
sys.path.insert(0, "/home/z/my-project/backend")
import autograd.numpy as anp
import numpy as np
import pennylane as qml

from qednet.sim.statevector import (BatchedCircuit, apply_gate_2x2,
                                    ry_matrix, ladder_entanglers)

n = 3
B = 4
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (B, n))
W = rng.uniform(0, 2 * np.pi, (2, n))  # 2 layers, 1 RY per qubit per layer

# --- my simulator ---
cir = BatchedCircuit(n)
state = cir.zero_state(B)
# angle encoding
for q in range(n):
    state = cir.ry(state, q, np.pi * X[:, q])
# trainable layers with entanglement
for l in range(2):
    for q in range(n):
        state = cir.ry(state, q, W[l, q])
    for c, t in ladder_entanglers(n):
        state = cir.cnot(state, c, t)
mine = cir.expval_z(state, 0)

# --- PennyLane reference ---
dev = qml.device("default.qubit", wires=n)

def circuit(x, w):
    for q in range(n):
        qml.RY(np.pi * x[q], wires=q)
    for l in range(2):
        for q in range(n):
            qml.RY(w[l, q], wires=q)
        for c, t in ladder_entanglers(n):
            qml.CNOT(wires=[c, t])
    return qml.expval(qml.PauliZ(0))

qnode = qml.QNode(circuit, dev)
pl = np.array([qnode(X[b], W) for b in range(B)])

print("mine:", np.array(mine))
print("PL:  ", pl)
print("max diff:", np.max(np.abs(np.array(mine) - pl)))
assert np.allclose(np.array(mine), pl, atol=1e-12), "MISMATCH"
print("OK: simulator matches PennyLane default.qubit")

# --- gradient check via autograd ---
from autograd import grad

def loss_fn(W):
    state = cir.zero_state(B)
    for q in range(n):
        state = cir.ry(state, q, np.pi * X[:, q])
    for l in range(2):
        for q in range(n):
            state = cir.ry(state, q, W[l, q])
        for c, t in ladder_entanglers(n):
            state = cir.cnot(state, c, t)
    o = cir.expval_z(state, 0)
    p = 1.0 / (1.0 + anp.exp(-2.0 * o))
    return -anp.mean(X[:, 0] * 0 + anp.log(p + 1e-12) * 0 + 0.5 * (o ** 2))

g = grad(loss_fn)(W)
print("grad norm:", float(np.linalg.norm(g)))
print("gate counts:", cir.gate_counts)
