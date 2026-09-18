"""Benchmark VQC training step speed on the batched simulator."""
import sys, time
sys.path.insert(0, "/home/z/my-project/backend")
import numpy as np
from autograd import grad
import autograd.numpy as anp

from qednet.models.quantum.vqc import VQC

X = np.random.default_rng(0).normal(size=(455, 8))
y = (X[:, 0] > 0).astype(int)

vqc = VQC(n_qubits=8, layers=3, epochs=1, batch_size=32)

# instrument: time one full fit with epochs=1
t0 = time.time()
vqc.fit(X, y)
t1 = time.time()
print(f"1 epoch on 455 samples (batch 32): {t1-t0:.2f}s")

t0 = time.time()
vqc.predict_proba(X[:114])
t1 = time.time()
print(f"predict 114 samples: {t1-t0:.3f}s")

# scale estimate
per_epoch = t1  # placeholder
from qednet.models.training import Adam, bce_loss, batches, sigmoid, quantum_input_transform
Xsq = quantum_input_transform(X)
rng = np.random.default_rng(42)
params = {"W": rng.uniform(0, 2*np.pi, (3, 8)), "alpha": np.array(2.0)}

def loss_b(pp, Xb, yb):
    out, alpha = vqc._circuit(Xb, pp["W"], pp["alpha"])
    return bce_loss(sigmoid(pp["alpha"] * out), yb)

t0 = time.time()
for bidx in batches(len(y), 32, rng):
    g = grad(lambda pp: loss_b(pp, Xsq[bidx], y[bidx].astype(float)))(params)
    params = Adam(params).step(params, g)
t1 = time.time()
print(f"one full epoch of grad steps (455/32=15 steps): {t1-t0:.2f}s")
