# QED-Net 2.0 Model Cards

Research artifacts — **not medical devices**. Metrics below are placeholders
for the *structure* of model cards; actual values are filled from each
experiment's `result.json` at generation time (the dashboard and reports
always show measured values).

---

## QSVM — Quantum Support Vector Machine

- **Task**: binary disease-risk classification (research)
- **Architecture**: ZZ / angle quantum feature map → fidelity kernel k(x,x') = |⟨φ(x)|φ(x')⟩|² → SVM (precomputed kernel, Platt probabilities)
- **Parameters**: none trainable (capacity = support vectors; reported post-fit)
- **Quantum resources**: n_qubits = target_dim (8 default), feature-map depth recorded per run; exact statevector fidelity (shots=∞ equivalent)
- **Data**: public/de-identified benchmark datasets only
- **Evaluation**: nested CV, multi-seed, AUROC/AUPRC/… + DeLong/bootstrap statistics
- **Limitations**: kernel scales O(N²) state overlaps — bounded by the 12-qubit resource limit; exact simulation (no shot noise)
- **Research status**: educational/research artifact; no clinical use

## VQC — Variational Quantum Classifier

- **Architecture**: RY angle encoding (π·squash(x)) → L layers × [RY(θ) per qubit + CNOT ladder] → Pauli-Z readout → sigmoid(α·⟨Z⟩)
- **Parameters**: L×n_qubits + 1 (e.g., 3×8+1 = 25)
- **Training**: Adam, BCE loss, early stopping; backprop through the exact statevector (batched simulator; PennyLane-verified equivalence)
- **Resources**: 8 qubits default; depth from executed gate schedule; circuit count = samples × epochs (logged)
- **Limitations**: real-unitary (RY/CNOT) ansatz — no RX/RZ phases; simulator-scale only

## Data Re-Uploading Classifier

- **Architecture**: per layer, per qubit: RY(φ) then RY(w·π·x + b) (feature re-injection with trainable scale/offset) + CNOT ring; mean Pauli-Z readout
- **Parameters**: 3×(L×n_qubits) + 1 (e.g., 73)
- **Reference**: Pérez-Salinas et al. (2020), simplified to real rotations

## Hybrid QNN

- **Architecture**: classical dense extractor (d→n_qubits, tanh) → angle-encoded RY-CNOT quantum layer → mean-Z readout → classical head
- **Parameters**: classical extractor + circuit + head (all trained jointly, end-to-end through the simulator)

## Classical baselines (deliberately strong)

- Logistic Regression (C tuned in inner CV)
- Random Forest (300 trees)
- **XGBoost** (300 rounds, depth tuned — treated as the strong tabular baseline)
- RBF-SVM (Platt probabilities)
- Parameter-Matched MLP (hidden layer sized to match the primary quantum model's parameter count — recorded in metadata)

## Calibration / uncertainty wrappers

- Platt & isotonic calibrators (fit on training-fold predictions only)
- Inductive conformal prediction (LAC) with empirical coverage reporting
- Abstention rule (confidence / ambiguity band / conformal set size) — never bypassed
