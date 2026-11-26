# QEDNET Quantum vs Classical Benchmark Results

## Results tables

Each table uses one small exploratory 80-row stratified sample and the same three shuffled stratified folds for the four classical models and the quantum models. Every model, including the classical models, receives only 4 PCA features. Each metric is mean +/- sample standard deviation across completed folds; the `n` in each cell is the number of completed folds used. Training time is estimator fit only; fold-local preprocessing and process startup are excluded. Values are written from Python's shortest round-trip float representation, without decimal rounding.

### breast_cancer

Rows: 80; input features: 30; class counts: 0=30, 1=50.
Fold sizes (train/test): [(53, 27), (53, 27), (54, 26)].

| Model | Completed folds | AUC | Accuracy | Balanced accuracy | F1 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 3/3 | 0.9862745098039216 +/- 0.018909119139201867 (n=3) | 0.9506172839506173 +/- 0.05657500857970175 (n=3) | 0.9539215686274509 +/- 0.05629363615668863 (n=3) | 0.9595959595959596 +/- 0.04628864338339234 (n=3) | 0.10645306666750305 +/- 0.07560528811246187 (n=3) |
| Random forest | 3/3 | 0.9823529411764707 +/- 0.025640582020827463 (n=3) | 0.9506172839506173 +/- 0.042766686606638946 (n=3) | 0.953921568627451 +/- 0.041211469719575865 (n=3) | 0.9595588235294118 +/- 0.035071294169740655 (n=3) | 1.6589364333328074 +/- 0.2511836388341193 (n=3) |
| RBF SVM | 3/3 | 0.9882352941176471 +/- 0.020377068324339723 (n=3) | 0.9506172839506173 +/- 0.05657500857970175 (n=3) | 0.9539215686274509 +/- 0.05629363615668863 (n=3) | 0.9595959595959596 +/- 0.04628864338339234 (n=3) | 0.08783923333370088 +/- 0.13806483012084658 (n=3) |
| sklearn MLP (early stopping enabled) | 3/3 | 0.5105392156862745 +/- 0.4248297117968267 (n=3) | 0.6870845204178537 +/- 0.1509187382633173 (n=3) | 0.603921568627451 +/- 0.17544637419777728 (n=3) | 0.7917415489944373 +/- 0.09785496658708956 (n=3) | 0.26079433333385776 +/- 0.1487218882770186 (n=3) |
| QSVM | 2/3; fold 1: timeout (2014.8350686000013 s elapsed) | 0.9669117647058825 +/- 0.011438492048605954 (n=2) | 0.9052706552706553 +/- 0.029210963895170795 (n=2) | 0.8955882352941176 +/- 0.06447150063759699 (n=2) | 0.9258928571428571 +/- 0.016414978848973444 (n=2) | 47.52374665000025 +/- 15.714751075295592 (n=2) |
| ReuploadClassifier | 3/3 | 0.8147058823529412 +/- 0.13110522992954515 (n=3) | 0.6885090218423552 +/- 0.07231829577776232 (n=3) | 0.5901960784313726 +/- 0.09622338005737226 (n=3) | 0.7982358998270871 +/- 0.0406281765779029 (n=3) | 329.48178069999875 +/- 75.87285848599078 (n=3) |
| VQC | 3/3 | 0.5370098039215686 +/- 0.1749448957567108 (n=3) | 0.5864197530864197 +/- 0.08350462505227581 (n=3) | 0.5280637254901961 +/- 0.04482329862659951 (n=3) | 0.6879452229570248 +/- 0.10073533618033849 (n=3) | 264.59863746666696 +/- 93.32914944485192 (n=3) |

Paired per-fold difference (best completed quantum AUC minus best classical AUC): folds [0, 1, 2]; (fold, best quantum model, best classical model) [(0, 'qsvm', 'rbf_svm'), (1, 'vqc', 'logistic_regression'), (2, 'qsvm', 'logistic_regression')]; differences [-0.04117647058823526, -0.23529411764705888, -0.02499999999999991]; mean -0.10049019607843135; sample standard deviation 0.11702346988054799. No clear difference.

### heart_disease

Rows: 80; input features: 13; class counts: 0=43, 1=37.
Fold sizes (train/test): [(53, 27), (53, 27), (54, 26)].

| Model | Completed folds | AUC | Accuracy | Balanced accuracy | F1 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 3/3 | 0.8681827431827432 +/- 0.10159131128156022 (n=3) | 0.774928774928775 +/- 0.07423825766834975 (n=3) | 0.7792735042735043 +/- 0.08228114130144003 (n=3) | 0.7664835164835164 +/- 0.08364471070130948 (n=3) | 0.035995699998845034 +/- 0.026240147852017535 (n=3) |
| Random forest | 3/3 | 0.84002849002849 +/- 0.073825411610907 (n=3) | 0.7497625830959164 +/- 0.024768853477198465 (n=3) | 0.7542735042735043 +/- 0.032565000447728684 (n=3) | 0.7419047619047618 +/- 0.03794016054674684 (n=3) | 1.603870033332593 +/- 0.022136894514582225 (n=3) |
| RBF SVM | 3/3 | 0.8709503459503459 +/- 0.12314364947228708 (n=3) | 0.8124406457739791 +/- 0.11118718023337522 (n=3) | 0.8114163614163614 +/- 0.11563730881799575 (n=3) | 0.7924297924297924 +/- 0.11837416349808418 (n=3) | 0.00644053333356472 +/- 0.003634773585201653 (n=3) |
| sklearn MLP (early stopping enabled) | 3/3 | 0.7301180301180301 +/- 0.12920825842492195 (n=3) | 0.6244064577397911 +/- 0.13599575165516672 (n=3) | 0.6371794871794871 +/- 0.13034619809669643 (n=3) | 0.6876802360673329 +/- 0.07064635525951891 (n=3) | 0.06913326666714663 +/- 0.03497996244917881 (n=3) |
| QSVM | 3/3 | 0.7436304436304436 +/- 0.1760763176187836 (n=3) | 0.6752136752136753 +/- 0.18577595882508385 (n=3) | 0.6728327228327229 +/- 0.186615597342599 (n=3) | 0.6388888888888888 +/- 0.20971762320196527 (n=3) | NOT MEASURED (conflicting independent wall-clock timings) |
| ReuploadClassifier | 3/3 | 0.37173382173382175 +/- 0.19823334619511496 (n=3) | 0.39886039886039887 +/- 0.08720927558412514 (n=3) | 0.40061050061050063 +/- 0.0941902384619887 (n=3) | 0.3782051282051282 +/- 0.12512321343553293 (n=3) | 204.66884393333507 +/- 92.4541292579997 (n=3) |
| VQC | 3/3 | 0.5883394383394384 +/- 0.02468544705255477 (n=3) | 0.4881291547958214 +/- 0.07061466835715595 (n=3) | 0.4843101343101343 +/- 0.061113619235226936 (n=3) | 0.4631093544137022 +/- 0.033441476011849956 (n=3) | 172.83308753333282 +/- 11.88702049110598 (n=3) |

Paired per-fold difference (best completed quantum AUC minus best classical AUC): folds [0, 1, 2]; (fold, best quantum model, best classical model) [(0, 'qsvm', 'rbf_svm'), (1, 'qsvm', 'logistic_regression'), (2, 'qsvm', 'rbf_svm')]; differences [-0.04999999999999993, -0.1923076923076924, -0.17261904761904756]; mean -0.1383089133089133; sample standard deviation 0.07710874664893606. Quantum worse on this paired comparison.

### parkinsons

Rows: 80; input features: 22; class counts: 0=20, 1=60.
Fold sizes (train/test): [(53, 27), (53, 27), (54, 26)].

| Model | Completed folds | AUC | Accuracy | Balanced accuracy | F1 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 3/3 | 0.8261904761904761 +/- 0.0607492896293956 (n=3) | 0.7749287749287749 +/- 0.03736432207579487 (n=3) | 0.7492063492063492 +/- 0.1186561022329983 (n=3) | 0.8428388043278928 +/- 0.013951752504390413 (n=3) | 0.136701966667412 +/- 0.16005021398832128 (n=3) |
| Random forest | 3/3 | 0.761904761904762 +/- 0.1258305739211792 (n=3) | 0.7625830959164293 +/- 0.019392761497305008 (n=3) | 0.6904761904761905 +/- 0.03647988915724899 (n=3) | 0.8393178893178893 +/- 0.024944844072420397 (n=3) | 1.4339585666675703 +/- 0.14305672197506614 (n=3) |
| RBF SVM | 3/3 | 0.8107142857142857 +/- 0.07977967108460243 (n=3) | 0.8262108262108262 +/- 0.08388850065148411 (n=3) | 0.7206349206349206 +/- 0.09759968836300814 (n=3) | 0.8888888888888888 +/- 0.0549857399228215 (n=3) | 0.010130599999683909 +/- 0.0036162907106932676 (n=3) |
| sklearn MLP (early stopping enabled) | 3/3 | 0.5448412698412698 +/- 0.1834659899363936 (n=3) | 0.6258309591642925 +/- 0.0684551615990127 (n=3) | 0.49801587301587297 +/- 0.02654877568837732 (n=3) | 0.7479114452798663 +/- 0.06268274310133215 (n=3) | 0.12758743333324674 +/- 0.01588361784077545 (n=3) |
| QSVM | 3/3 | 0.7797619047619048 +/- 0.11792927085829905 (n=3) | 0.7625830959164293 +/- 0.019392761497305008 (n=3) | 0.5392857142857143 +/- 0.036246041303900765 (n=3) | 0.8611916264090177 +/- 0.01450348502312102 (n=3) | 31.327701233332238 +/- 3.59818750674024 (n=3) |
| ReuploadClassifier | 3/3 | 0.5976190476190476 +/- 0.10359195198955763 (n=3) | 0.6885090218423552 +/- 0.07231829577776232 (n=3) | 0.6404761904761905 +/- 0.03240632759241953 (n=3) | 0.7763263263263264 +/- 0.06609853374554125 (n=3) | 218.139863666666 +/- 108.62776192238982 (n=3) |
| VQC | 2/3; fold 2: timeout (901.6490694999993 s elapsed) | 0.5321428571428571 +/- 0.06565991539589362 (n=2) | 0.5555555555555556 +/- 0.05237828008789241 (n=2) | 0.5375 +/- 0.002525381361380518 (n=2) | 0.6543816543816544 +/- 0.06833628188371223 (n=2) | 207.92466739999873 +/- 18.233725574465605 (n=2) |

Paired per-fold difference (best completed quantum AUC minus best classical AUC): folds [0, 1, 2]; (fold, best quantum model, best classical model) [(0, 'qsvm', 'logistic_regression'), (1, 'qsvm', 'logistic_regression'), (2, 'qsvm', 'rbf_svm')]; differences [0.02857142857142858, -0.09285714285714275, -0.09999999999999998]; mean -0.05476190476190471; sample standard deviation 0.0722570995499691. No clear difference. Parkinson's uses random stratified folds, not subject-grouped folds; overlapping train/test subjects by fold: 14 overlap_ids=['S01', 'S02', 'S06', 'S07', 'S16', 'S20', 'S21', 'S22', 'S32', 'S34', 'S35', 'S39', 'S42', 'S43']; 16 overlap_ids=['S01', 'S05', 'S06', 'S07', 'S16', 'S18', 'S19', 'S20', 'S22', 'S25', 'S27', 'S32', 'S34', 'S42', 'S44', 'S49']; 17 overlap_ids=['S02', 'S05', 'S16', 'S18', 'S19', 'S20', 'S21', 'S22', 'S25', 'S27', 'S32', 'S35', 'S39', 'S42', 'S43', 'S44', 'S49'].

### diabetes

Rows: 80; input features: 8; class counts: 0=52, 1=28.
Fold sizes (train/test): [(53, 27), (53, 27), (54, 26)].

| Model | Completed folds | AUC | Accuracy | Balanced accuracy | F1 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 3/3 | 0.8222222222222222 +/- 0.07489865281451916 (n=3) | 0.7003798670465337 +/- 0.09626018529807959 (n=3) | 0.6919389978213507 +/- 0.14021143986005294 (n=3) | 0.5864293085655314 +/- 0.16448112045351823 (n=3) | 0.04321110000091721 +/- 0.02320306628275388 (n=3) |
| Random forest | 3/3 | 0.784870491406439 +/- 0.054378837334201215 (n=3) | 0.7136752136752137 +/- 0.11007442448405415 (n=3) | 0.6880174291938997 +/- 0.15264808144661932 (n=3) | 0.5560670645069622 +/- 0.2506802881028501 (n=3) | 0.6091315666672017 +/- 0.6193019741340832 (n=3) |
| RBF SVM | 3/3 | 0.8098039215686275 +/- 0.049487958682839055 (n=3) | 0.738366571699905 +/- 0.07054279102537049 (n=3) | 0.6946623093681917 +/- 0.07312257334325296 (n=3) | 0.5842131724484666 +/- 0.10168791449046952 (n=3) | 0.007742033331548252 +/- 0.003733107569772715 (n=3) |
| sklearn MLP (early stopping enabled) | 3/3 | 0.4255628177196805 +/- 0.17753095094184107 (n=3) | 0.3765432098765432 +/- 0.10850861624226511 (n=3) | 0.41470588235294115 +/- 0.13226738136607064 (n=3) | 0.3741690408357075 +/- 0.12514010567923367 (n=3) | 0.24812979999963622 +/- 0.20991411622520237 (n=3) |
| QSVM | 3/3 | 0.5938029532800775 +/- 0.12222479681855238 (n=3) | 0.6011396011396012 +/- 0.07895388296610265 (n=3) | 0.4872549019607843 +/- 0.05978070194614715 (n=3) | 0.1619047619047619 +/- 0.03299144395369291 (n=3) | 40.374599333333514 +/- 12.432102513565031 (n=3) |
| ReuploadClassifier | 3/3 | 0.7186153473735173 +/- 0.09059285111997277 (n=3) | 0.6894586894586894 +/- 0.18766691231195962 (n=3) | 0.6610021786492375 +/- 0.22704454646861014 (n=3) | 0.5231481481481481 +/- 0.34925331875078325 (n=3) | 343.7283201999987 +/- 19.550841237501757 (n=3) |
| VQC | 3/3 | 0.5741466957153232 +/- 0.10085059646120688 (n=3) | 0.5508072174738842 +/- 0.09281513179768751 (n=3) | 0.5362745098039216 +/- 0.054559506188143266 (n=3) | 0.43187830687830686 +/- 0.051752381374311186 (n=3) | 203.3044444333306 +/- 54.02405594959997 (n=3) |

Paired per-fold difference (best completed quantum AUC minus best classical AUC): folds [0, 1, 2]; (fold, best quantum model, best classical model) [(0, 'reupload', 'logistic_regression'), (1, 'reupload', 'rbf_svm'), (2, 'reupload', 'rbf_svm')]; differences [-0.16049382716049387, -0.12941176470588234, -0.03921568627450989]; mean -0.1097070927136287; sample standard deviation 0.06299446354983805. Quantum worse on this paired comparison.

### Plain-language AUC statements

- On breast_cancer, RBF SVM had the highest mean AUC among model rows; quantum models were numerically worse, but no clear difference on the paired fold comparison.
- On heart_disease, RBF SVM had the highest mean AUC among model rows; quantum models were worse on the paired fold comparison.
- On parkinsons, Logistic regression had the highest mean AUC among model rows; quantum models were numerically worse, but no clear difference on the paired fold comparison.
- On diabetes, Logistic regression had the highest mean AUC among model rows; quantum models were worse on the paired fold comparison.

## Quantum model configuration and run status

All quantum models used PennyLane `default.qubit` (simulator, not real hardware), 4 qubits, and `shots=None` (analytic expectation values). The input pipeline reduced each fold's training data to 4 PCA features after imputation, scaling, and SelectKBest feature selection.

| Model | Variational layers / kernel depth | Trainable parameters | Feature map / encoding | Device | Shots |
|---|---:|---:|---|---|---|
| QSVM | no trainable layers; feature-map circuit depth=3 | 0 | zz_feature_map | default.qubit | analytic (`shots=None`) |
| ReuploadClassifier | 1 | 16 | RY data reupload with trainable offsets | default.qubit | analytic (`shots=None`) |
| VQC | 1 | 12 | angle encoding | default.qubit | analytic (`shots=None`) |

### Timing gate and QSVM kernel evaluations

Heart disease timing fold: n=80, train=53, test=27, seed=42. QSVM fit 53.85604339999918 s; QSVM `predict_proba` 48.33107899999959 s; fit kernel evaluations 1431. ReuploadClassifier first full-run fold fit 138.50894860000335 s and prediction 0.3275616999999329 s. VQC first full-run fold fit 164.46560800000225 s and prediction 0.2712131000007503 s.
Timing discrepancy: the separate gate run recorded QSVM fold-zero fit=53.85604339999918 s and prediction=48.33107899999959 s; the shared-grid run recorded fit=15.134612500001822 s and prediction=25.25466719999895 s. Both record n_train=53, n_test=27, and 1431 fit-kernel evaluations with the same seed/configuration. Three further direct repeats produced fit times [65.45583069999702, 36.406415699995705, 27.257883699996455] s and prediction times [111.80249069999991, 135.01713529999688, 31.62832769999659] s. Across all five observations, fit times ranged from 15.134612500001822 to 65.45583069999702 s. Same config, fold size, and kernel counts were confirmed; the underlying cause of wall-clock variation is NOT MEASURED. The heart QSVM training-time cell is therefore NOT MEASURED as a reliable summary.
QSVM fit kernel evaluation counts on breast_cancer: [(0, 1431), (2, 1485)].
QSVM fit kernel evaluation counts on heart_disease: [(0, 1431), (1, 1431), (2, 1485)].
QSVM fit kernel evaluation counts on parkinsons: [(0, 1431), (1, 1431), (2, 1485)].
QSVM fit kernel evaluation counts on diabetes: [(0, 1431), (1, 1431), (2, 1485)].

Quantum grid status: 34 completed fold records out of 36 expected model-fold records. HybridQNN was not run because it delegates to a random-forest mock. The optional seed 1729 repeat is NOT MEASURED; this execution only ran the seed-42 grid. Full-dataset quantum results are NOT MEASURED.

Runs that did not finish:
- breast_cancer / qsvm / fold 1 / seed 42: status `timeout`, elapsed 2014.8350686000013 s, configured timeout 900 s.
- parkinsons / vqc / fold 2 / seed 42: status `timeout`, elapsed 901.6490694999993 s, configured timeout 900 s.

## MLP diagnosis

The 80-row results table still uses the broken `MLPClassifier(early_stopping=True)` baseline; those 80-row AUCs are not replaced by the separate rerun. The diagnosis uses a different, existing 150-row stratified sample and the same 3 folds (seed 42; 100 training and 50 test rows per fold). The results-only rerun used those same 150-row data and folds with early stopping disabled. No model or pipeline source was edited.

| Dataset | Early stopping AUC | Early stopping accuracy | Early stopping disabled AUC | Early stopping disabled accuracy |
|---|---:|---:|---:|---:|
| breast_cancer | 0.46518247343268565 +/- 0.4724043091579325 (n=3) | 0.6533333333333333 +/- 0.19731531449264988 (n=3) | 0.9864176570458404 +/- 0.01928322018947463 (n=3) | 0.9533333333333334 +/- 0.030550504633038898 (n=3) |
| heart_disease | 0.45356951154052605 +/- 0.11527286467468426 (n=3) | 0.4066666666666667 +/- 0.07571877794400365 (n=3) | 0.8663446054750402 +/- 0.021302345499714912 (n=3) | 0.7933333333333333 +/- 0.06429100507328636 (n=3) |
| parkinsons | 0.5908058868585184 +/- 0.2464813720351754 (n=3) | 0.7066666666666667 +/- 0.08082903768654762 (n=3) | 0.9013264276422172 +/- 0.028998371838217664 (n=3) | 0.84 +/- 0.020000000000000018 (n=3) |
| diabetes | 0.5198275648643296 +/- 0.18360043141524282 (n=3) | 0.4 +/- 0.08717797887081348 (n=3) | 0.8406708011487423 +/- 0.03154978282219687 (n=3) | 0.7533333333333333 +/- 0.023094010767585053 (n=3) |

One-fold diagnostic (breast cancer, fold 0): preprocessing produced 4 features and the transformed test values matched the fitted training-only preprocessing transform (`preprocess_test_equal=True`). Training labels were `{0: 38, 1: 62}` and test labels `{0: 18, 1: 32}`. `classes_` was `[0, 1]`, and positive-class probabilities used column index `1`. With early stopping enabled, the fold stopped at `n_iter=12` with AUC `0.9982638888888888`; with early stopping disabled it reached `n_iter=300` with AUC `1.0`. The printed probabilities and predicted labels are saved in `results/diagnose_mlp_onefold_raw.txt`.

The evidence points to unstable early stopping on the small training folds as the results-script setup issue: fold-to-fold AUCs varied sharply with early stopping, while the same folds without early stopping recovered substantially higher AUC. The evidence does not indicate a reversed positive-probability column or a test preprocessing mismatch. This diagnosis is limited to these data, folds, and MLP configuration.

## How measured

- Branch: `results`.
- Dataset sources: breast cancer from scikit-learn; heart disease from `data/raw/heart_cleveland.data`; Parkinson's from `data/raw/parkinsons.data`; diabetes from `data/raw/pima_diabetes.csv`.
- Each shared sample was recreated from the existing 150-row stratified sample using `StratifiedShuffleSplit(train_size=80, random_state=42)`. The shared folds used `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`. Fold sizes are printed above. Quantum and classical models used the exact same manifest rows and split indices. This is a single fixed subsample and a single 3-fold split, not repeated or nested cross-validation.
- Parkinson's 80-row benchmark uses random stratified folds, not `StratifiedGroupKFold`. The sample contains repeated subject IDs across train and test; overlap counts per fold were measured by `python -X utf8 results/audit_parkinsons_80.py` and saved in `results/audit_parkinsons_80_raw.txt`. Grouped 80-row quantum results are NOT MEASURED.
- Base subsample and fold seed: 42. Model/preprocessing seeds per fold: 42, 43, and 44 (`base_seed + fold`). The optional seed 1729 was NOT MEASURED because no second-seed grid was run in this execution.
- Fold-local preprocessing: median imputation, `StandardScaler`, `SelectKBest(mutual_info_classif, k=8)`, then `PCA(n_components=4)`. The pipeline is fit on each training fold and applied with `transform` to its held-out fold. No fitting or feature selection uses held-out samples.
- Full preprocessing-plus-estimator training time is NOT MEASURED: the scripts start the reported training timer after fold-local preprocessing has completed. Reported training seconds are estimator-fit time only.
- Classical command: `python results/run_shared80_classical.py`; saved exact command and output: `results/shared80_classical_command.txt`, `results/shared80_classical_stdout.txt`; fold records: `results/shared80_classical_raw.json`.
- Quantum command: `python results/run_quantum_shared80.py`; each fold was launched in a child process with a configured 900-second timeout and wrote a raw stdout/stderr file. The runner failed to enforce that limit for breast-cancer QSVM fold 1: it recorded 2014.8350686000013 seconds before marking timeout. Do not claim all timeouts stopped at 900 seconds. The parent appended each completed or timed-out fold to `results/quantum_raw.json` immediately. Exact command and runner output: `results/quantum_shared80_command.txt`, `results/quantum_shared80_runner_stdout.txt`.
- Summary command: `python -X utf8 results/summarize_shared80.py`; output: `results/shared80_summary_raw.txt`.
- Per-fold CSV command: `python -X utf8 results/export_shared80_per_fold.py`; output: `results/shared80_per_fold.csv` (all model/fold records plus per-fold best-model and paired-difference columns). Its sample rows and column/source audit are in `results/audit_csvs_raw.txt`.
- Parkinson's subject-overlap audit command: `python -X utf8 results/audit_parkinsons_80.py`; output: `results/audit_parkinsons_80_raw.txt`.
- QSVM timing consistency commands: `python -X utf8 results/repeat_qsvm_timing80.py` and `python -X utf8 results/audit_qsvm_timing_consistency.py`; per-repeat raw data: `results/qsvm_timing_repeats80.json`; audit: `results/audit_qsvm_timing_consistency_raw.txt`.
- MLP commands and outputs: `results/diagnose_mlp_onefold_command.txt`, `results/diagnose_mlp_onefold_raw.txt`, `results/rerun_mlp_no_early_stopping_command.txt`, and `results/mlp_no_early_stopping_raw.json`. The existing classical early-stopping rows are in `results/shared150_classical_raw.json`.
- The 150-row heart-disease timing-gate attempt did not complete (`EXIT_CODE=-1`, stdout empty); no timing result is claimed from that attempt. The 80-row heart-disease QSVM timing result is saved in `results/quantum_timing80.json`.
- Classical timing and the shared-grid quantum timing record use the same timer placement, after preprocessing. Independent heart QSVM runs with matching fold and kernel counts show substantial wall-clock variation; the cause is NOT MEASURED.

## Limitations

- Results use 80 rows per dataset, not the full datasets, with one subsample seed and one set of three folds. They are exploratory and do not establish generalization or statistical significance.
- The classical models are restricted to 4 PCA features to match quantum input size. This controls feature count but may handicap classical models compared with fitting them on the original input features; full-feature baselines are NOT MEASURED in this comparison.
- Parkinson's random folds leak subjects: the same subject IDs occur in train and test. No Parkinson's quantum result here is subject-grouped. The full-195-row grouped result applies only to the earlier classical models.
- Two quantum model-folds are marked timeout. The configured 900-second timeout was not enforced as expected for breast-cancer QSVM fold 1: the recorded elapsed time is 2014.8350686000013 seconds. The runner timeout enforcement is broken for that run. Per-model metric means use only completed folds; the fold count is shown in each cell.
- Paired fold differences use the best completed quantum score available on each fold and the best classical score on that same fold. The winning quantum model can change by fold; this is not a fixed-model paired comparison. The fold-level winners and differences are in `results/shared80_per_fold.csv`.
- End-to-end preprocessing-plus-fit duration is NOT MEASURED because the timer begins after preprocessing; the reported training times cover estimator `.fit` only.
- The heart-disease QSVM training-time summary is NOT MEASURED as a reliable runtime statistic: repeated same-fold runs differed despite matching seed, split sizes, and kernel evaluation counts. The underlying cause is NOT MEASURED.
- Quantum results are from the PennyLane `default.qubit` simulator with analytic shots, not real quantum hardware.
- Seed 1729 is NOT MEASURED; no repeat run was made.
- `ParameterMatchedMLP` exists in the repository but is NOT MEASURED in this comparison; the requested shared-sample table covers the four existing classical baselines and the finished quantum models. The prior full-data attempt did not finish.
- Subject-grouped Parkinson's quantum results are NOT MEASURED; the reported 80-row Parkinson's scores use random stratified folds with train/test subject overlap.
- SHAP is NOT MEASURED. Nested cross-validation is NOT MEASURED; this run used a single stratified 3-fold split without nested tuning.
- HybridQNN is NOT MEASURED because its `fit` delegates to a random-forest mock.
- Calibration and conformal prediction are NOT MEASURED in this quantum-only task.
- MLP early-stopping behavior may vary with different samples or configurations; the diagnosis does not establish a universal MLP failure.

## Safe to claim on a resume

- Benchmarked four 4-feature PCA classical baselines against three PennyLane quantum classifiers on one small exploratory, 80-row stratified sample per dataset, using 3-fold CV with fold-local preprocessing.
- On these small exploratory samples, the best classical model had the highest mean AUC on every dataset; quantum was worse on heart disease and diabetes and showed no clear difference on breast cancer and Parkinson's. Parkinson's uses random folds with subject overlap.
- Found that early stopping made an MLP baseline unstable on small folds; disabling early stopping improved it on a separate 150-row sample, not in the 80-row benchmark table.
- Reported 34 of 36 quantum fits completing; 2 were marked timeout, and one ran over the configured limit because of a runner bug.

## NOT safe to claim

- That quantum models outperform classical models generally, on the full datasets, or on real hardware.
- That any observed difference is statistically significant or robust across random seeds.
- That all requested quantum fits completed within the configured timeout; the runner failed to enforce it for one fold.
- That the MLP issue is fixed in model/pipeline source; only the results-script configuration was investigated.
- Subject-grouped Parkinson's quantum results, seed 1729, ParameterMatchedMLP, HybridQNN, calibration, conformal prediction, SHAP, nested CV, or full-dataset quantum results.
