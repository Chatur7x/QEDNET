# QEDNET fixed benchmark v2 results

## Summary tables

Each metric is the mean ± sample standard deviation across completed folds, with the observed fold count. Values are printed at the precision stored in the current-run analysis output; no rounding is applied here. `Fit seconds` measures estimator `.fit` only, excluding preprocessing. A = median imputation, scaling, SelectKBest(8), PCA(4); B = the same fold-local imputation and scaling with all original input features (no selector or PCA). Quantum models have no setting B. ParameterMatchedMLP training times use the same estimator-only timing convention.

| Dataset and split | Rows; original features; class counts | Folds (train/test) |
| --- | --- | --- |
| breast_cancer (random) | 80; 30; 0:30, 1:50 | 53/27; 53/27; 54/26 |
| heart_disease (random) | 80; 13; 0:43, 1:37 | 53/27; 53/27; 54/26 |
| parkinsons (random) | 80; 22; 0:20, 1:60 | 53/27; 53/27; 54/26 |
| parkinsons (grouped) | 80; 22; 0:20, 1:60 | 53/27; 53/27; 54/26 |
| diabetes (random) | 80; 8; 0:52, 1:28 | 53/27; 53/27; 54/26 |

For every table below, rows were sampled directly from each complete dataset using `StratifiedShuffleSplit(train_size=80, random_state=42)`. The same fixed three folds were reused for all models and estimator seeds; the seed varies model initialization, not fold membership. Per-seed and per-fold raw records are in `classical_raw.json`, `quantum_raw.json`, and `summary.csv`.

### breast cancer — random folds — setting A

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.9803921568627451 ± 0.015563243006262306 (n=9) | 0.9377967711301045 ± 0.017806267806267817 (n=9) | 0.9372549019607843 ± 0.013070874143111113 (n=9) | 0.9494578134284016 ± 0.015262618398660614 (n=9) | 0.021033933333380144 ± 0.012901393391200396 (n=9) | 5 |
| mlp_default | 9/0/0/0 | 0.9777777777777777 ± 0.016838788271880074 (n=9) | 0.9629629629629629 ± 0.032075014954979206 (n=9) | 0.9637254901960784 ± 0.02747285542980796 (n=9) | 0.9696428571428571 ± 0.027096412331238426 (n=9) | 0.42125601111112904 ± 0.3598221976159699 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 9/0/0/0 | 0.7967320261437909 ± 0.22787686210532235 (n=9) | 0.5416270971826528 ± 0.12508423996010173 (n=9) | 0.5 ± 0.0 (n=9) | 0.5127465127465127 ± 0.38458526484529926 (n=9) | 4.47598109999995 ± 1.4882456808674278 (n=9) | 15 |
| parameter_matched_mlp_vqc | 9/0/0/0 | 0.7764705882352941 ± 0.2235342491736629 (n=9) | 0.7166824944602722 ± 0.13941285569510323 (n=9) | 0.6245098039215686 ± 0.1881663476673331 (n=9) | 0.8210253798489092 ± 0.07952619812759348 (n=9) | 5.0003103888888445 ± 0.917534757449783 (n=9) | 12 |
| qsvm | 9/0/0/0 | 0.939093137254902 ± 0.05433252626619829 (n=9) | 0.8632478632478633 ± 0.04765164867144657 (n=9) | 0.8372549019607843 ± 0.05611407067158508 (n=9) | 0.8961314255431903 ± 0.03462206953955891 (n=9) | 6.124928455555265 ± 0.3029804954620155 (n=9) | 0 |
| random_forest | 9/0/0/0 | 0.9581699346405229 ± 0.03619493125877399 (n=9) | 0.9377967711301045 ± 0.017806267806267817 (n=9) | 0.9372549019607843 ± 0.013070874143111113 (n=9) | 0.9494578134284016 ± 0.015262618398660614 (n=9) | 1.5307536000000508 ± 0.7815794642701469 (n=9) | 3242, 3378, 3586, 4160, 4224, 4370, 4686, 4724, 4892 |
| rbf_svm | 9/0/0/0 | 0.9745098039215686 ± 0.022971322576196055 (n=9) | 0.9588477366255144 ± 0.0343689158199384 (n=9) | 0.9604575163398693 ± 0.03541984686176493 (n=9) | 0.9667137056842939 ± 0.02773000095793527 (n=9) | 0.010724999999916286 ± 0.007809981555403035 (n=9) | NOT MEASURED |
| reupload | 9/0/0/0 | 0.8009803921568628 ± 0.07305268249979069 (n=9) | 0.7169990503323836 ± 0.06117068533784583 (n=9) | 0.6290849673202614 ± 0.07704036100736134 (n=9) | 0.8132318148578311 ± 0.03661860531503132 (n=9) | 50.337041844444684 ± 1.4837914033855535 (n=9) | 16 |
| vqc | 9/0/0/0 | 0.7235702614379085 ± 0.17365569040302758 (n=9) | 0.6327951883507439 ± 0.12166521851037077 (n=9) | 0.5733251633986928 ± 0.09379164139915433 (n=9) | 0.7177589054422995 ± 0.1367821839591741 (n=9) | 38.38536309999957 ± 1.4972875502255352 (n=9) | 12 |

### breast cancer — random folds — setting B

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.9859068627450981 ± 0.010770552996536911 (n=9) | 0.9624881291547958 ± 0.0007122507122506727 (n=9) | 0.9568627450980391 ± 0.010294117647058842 (n=9) | 0.9702741702741703 ± 0.0008658008658008476 (n=9) | 0.04695054444437119 ± 0.022227282023232455 (n=9) | 31 |
| mlp_default | 9/0/0/0 | 0.9873366013071896 ± 0.011168109402572802 (n=9) | 0.9624881291547958 ± 0.0007122507122506727 (n=9) | 0.9568627450980391 ± 0.010294117647058842 (n=9) | 0.9702741702741703 ± 0.0008658008658008476 (n=9) | 0.3223296444444309 ± 0.15392978934999607 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| parameter_matched_mlp_vqc | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| random_forest | 9/0/0/0 | 0.9738562091503268 ± 0.031002722158513542 (n=9) | 0.9544159544159544 ± 0.024580390209027626 (n=9) | 0.9526143790849674 ± 0.028074158947600666 (n=9) | 0.9635571400277283 ± 0.019426955109057426 (n=9) | 2.413265199999943 ± 2.024903866127336 (n=9) | 2642, 2660, 2696, 3166, 3194, 3204, 3504, 3508 |
| rbf_svm | 9/0/0/0 | 0.9862745098039216 ± 0.010604562574894049 (n=9) | 0.9382716049382716 ± 0.06676946806414794 (n=9) | 0.9441176470588235 ± 0.05113334897268739 (n=9) | 0.946031746031746 ± 0.06079593016573191 (n=9) | 0.03870117777776411 ± 0.034083329700333266 (n=9) | NOT MEASURED |

### heart disease — random folds — setting A

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.8168633835300502 ± 0.05725800717384856 (n=9) | 0.7500791389680278 ± 0.051858069540835924 (n=9) | 0.7431318681318682 ± 0.06354443707860853 (n=9) | 0.7064124551480874 ± 0.11502467749242425 (n=9) | 0.07153511111128966 ± 0.1375646931522988 (n=9) | 5 |
| mlp_default | 9/0/0/0 | 0.7661884411884412 ± 0.06377112481027535 (n=9) | 0.6918328584995251 ± 0.05037380259284025 (n=9) | 0.68491045991046 ± 0.05973040678490612 (n=9) | 0.6409515327643982 ± 0.10739906184160372 (n=9) | 0.4292120000000068 ± 0.15764037104015924 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 9/0/0/0 | 0.624859245692579 ± 0.13559559955297742 (n=9) | 0.4874960430515986 ± 0.04080256611384774 (n=9) | 0.5 ± 0.0 (n=9) | 0.42154745838956365 ± 0.3163977019017836 (n=9) | 6.468942066666589 ± 2.8233583296032645 (n=9) | 15 |
| parameter_matched_mlp_vqc | 9/0/0/0 | 0.619459706959707 ± 0.13302850328942734 (n=9) | 0.5794555238999683 ± 0.13176402896524558 (n=9) | 0.5943630443630443 ± 0.12159733553370879 (n=9) | 0.5860642521360762 ± 0.23591978839686173 (n=9) | 11.069124566666789 ± 15.166646215302166 (n=9) | 12 |
| qsvm | 9/0/0/0 | 0.6813932980599647 ± 0.1050475752658384 (n=9) | 0.6449825894270339 ± 0.12257703931823402 (n=9) | 0.6353174603174603 ± 0.12599216795381454 (n=9) | 0.5688196666457536 ± 0.15296736983159023 (n=9) | 8.018959766666537 ± 3.9284810318743806 (n=9) | 0 |
| random_forest | 9/0/0/0 | 0.8161036494369828 ± 0.07913060324088159 (n=9) | 0.7209559987337765 ± 0.06389029608246645 (n=9) | 0.712006512006512 ± 0.07748131434634208 (n=9) | 0.6605661983822904 ± 0.14544616143366035 (n=9) | 3.7624701333334087 ± 6.826440170078434 (n=9) | 6620, 6646, 6662, 6756, 6820, 7048, 7084, 7142, 7156 |
| rbf_svm | 9/0/0/0 | 0.8031372948039615 ± 0.05491595683889003 (n=9) | 0.7082937638493194 ± 0.06127748231719137 (n=9) | 0.7013634513634514 ± 0.06073269539448832 (n=9) | 0.6661405261405261 ± 0.07896847108264357 (n=9) | 0.022010866666581325 ± 0.026162374762332435 (n=9) | NOT MEASURED |
| reupload | 9/0/0/0 | 0.5882342965676299 ± 0.0942099832081913 (n=9) | 0.5419436530547641 ± 0.0792236409961567 (n=9) | 0.5427248677248677 ± 0.07937850630762971 (n=9) | 0.5214891599272938 ± 0.10956561980076533 (n=9) | 69.41908447777799 ± 26.84127206128184 (n=9) | 16 |
| vqc | 9/0/0/0 | 0.631396689730023 ± 0.1022748437452225 (n=9) | 0.5751820196264641 ± 0.09027838263472807 (n=9) | 0.5801689051689052 ± 0.09108213370019387 (n=9) | 0.5699175985716587 ± 0.11408184777532196 (n=9) | 44.218291000000235 ± 9.580374429809995 (n=9) | 12 |

### heart disease — random folds — setting B

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.8350427350427351 ± 0.04560807317063611 (n=9) | 0.7621082621082621 ± 0.06832040112888868 (n=9) | 0.7566239316239317 ± 0.076313829354792 (n=9) | 0.7257142857142856 ± 0.11142857142857142 (n=9) | 0.007354833333718286 ± 0.0014509946976512447 (n=9) | 14 |
| mlp_default | 9/0/0/0 | 0.7899267399267399 ± 0.07987218970367772 (n=9) | 0.7204811649256093 ± 0.09033736598861793 (n=9) | 0.7115181115181115 ± 0.09978605165742764 (n=9) | 0.6608339649319159 ± 0.14790138450039952 (n=9) | 0.18525306666672098 ± 0.041791278615866664 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 9/0/0/0 | 0.7976546601546601 ± 0.07942405881194187 (n=9) | 0.7453308008863564 ± 0.08484391286650844 (n=9) | 0.7447700447700447 ± 0.08238462716746722 (n=9) | 0.7317869276040191 ± 0.08628222884936727 (n=9) | 3.6850975111107496 ± 2.449000644900039 (n=9) | 16 |
| parameter_matched_mlp_vqc | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| random_forest | 9/0/0/0 | 0.844047619047619 ± 0.04790150080336682 (n=9) | 0.7831592276036721 ± 0.0555088642357481 (n=9) | 0.773951973951974 ± 0.06644389957394942 (n=9) | 0.7341164358708219 ± 0.10752581430066945 (n=9) | 0.8759515444444332 ± 0.37593324769750713 (n=9) | 7314, 7344, 7414, 7732, 7764, 7828, 8094, 8110, 8168 |
| rbf_svm | 9/0/0/0 | 0.810958485958486 ± 0.041924218095613044 (n=9) | 0.7503956948401392 ± 0.06526244792966886 (n=9) | 0.7404456654456655 ± 0.07688080722021323 (n=9) | 0.6929204760420731 ± 0.13003401479588475 (n=9) | 0.0035630777777745 ± 0.0012765901613306842 (n=9) | NOT MEASURED |

### Parkinson's — random folds — setting A

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.9177248677248677 ± 0.0628038093440835 (n=9) | 0.8285849952516618 ± 0.04868378202460129 (n=9) | 0.8317460317460318 ± 0.07974857742246586 (n=9) | 0.878595194384668 ± 0.03192344098837125 (n=9) | 0.007561222222041882 ± 0.002188259541137377 (n=9) | 5 |
| mlp_default | 9/0/0/0 | 0.907010582010582 ± 0.06447954806697737 (n=9) | 0.8741690408357075 ± 0.05237343719757759 (n=9) | 0.8003968253968254 ± 0.044180141700032205 (n=9) | 0.918136127438453 ± 0.03605487300112278 (n=9) | 0.18383477777762083 ± 0.03496643831629048 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 9/0/0/0 | 0.5863095238095238 ± 0.13195570724098823 (n=9) | 0.5834124723013612 ± 0.2506425447717701 (n=9) | 0.5 ± 0.0 (n=9) | 0.5714873059923938 ± 0.4286820260822748 (n=9) | 2.332776322222142 ± 0.24240947132994708 (n=9) | 15 |
| parameter_matched_mlp_vqc | 9/0/0/0 | 0.5783730158730158 ± 0.14093133992727147 (n=9) | 0.7374169040835707 ± 0.03367974477930738 (n=9) | 0.49166666666666664 ± 0.025000000000000005 (n=9) | 0.8484678986582909 ± 0.02309768839815375 (n=9) | 2.3925152555554328 ± 0.2819207909146418 (n=9) | 12 |
| qsvm | 9/0/0/0 | 0.7203703703703703 ± 0.17629043621601984 (n=9) | 0.795346628679962 ± 0.07135316117588714 (n=9) | 0.611904761904762 ± 0.13233978248134942 (n=9) | 0.8787079103754017 ± 0.0394230121068836 (n=9) | 11.721235133333217 ± 4.1935530472158 (n=9) | 0 |
| random_forest | 9/0/0/0 | 0.8634920634920635 ± 0.09222733607382565 (n=9) | 0.8200379867046533 ± 0.08072035080630877 (n=9) | 0.7667989417989418 ± 0.07742316715815233 (n=9) | 0.8773418589208063 ± 0.05682889254311859 (n=9) | 0.8893376111110329 ± 0.4193839005941754 (n=9) | 5286, 5314, 5376, 5418, 5934, 5944, 5996, 6040, 6164 |
| rbf_svm | 9/0/0/0 | 0.8923280423280423 ± 0.0821301558311598 (n=9) | 0.836815447926559 ± 0.0553868755213724 (n=9) | 0.7572751322751323 ± 0.059736063364720746 (n=9) | 0.8934153890478337 ± 0.03698108762656037 (n=9) | 0.004051555555633968 ± 0.0016863562116068939 (n=9) | NOT MEASURED |
| reupload | 9/0/0/0 | 0.646957671957672 ± 0.14856302335130847 (n=9) | 0.7238050015827794 ± 0.11634663771441195 (n=9) | 0.5904761904761905 ± 0.12279950890681568 (n=9) | 0.8199223436886511 ± 0.08392595678610287 (n=9) | 84.90340650000002 ± 28.065042079945176 (n=9) | 16 |
| vqc | 9/0/0/0 | 0.5735449735449736 ± 0.15499270539325288 (n=9) | 0.6405508072174739 ± 0.1902389519536719 (n=9) | 0.5387566137566138 ± 0.16811984583884734 (n=9) | 0.7387859861855558 ± 0.16684224235904782 (n=9) | 69.07072964444443 ± 21.715501937712983 (n=9) | 12 |

### Parkinson's — random folds — setting B

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.8472222222222222 ± 0.019019696796036285 (n=9) | 0.7882241215574549 ± 0.046312729934498184 (n=9) | 0.7420634920634921 ± 0.0323132608289047 (n=9) | 0.8536036036036037 ± 0.03871450635345077 (n=9) | 0.008099344444316457 ± 0.00200640056821354 (n=9) | 23 |
| mlp_default | 9/0/0/0 | 0.903042328042328 ± 0.05930349439083642 (n=9) | 0.8795504906616017 ± 0.05096008811285238 (n=9) | 0.8276455026455026 ± 0.05747601841962995 (n=9) | 0.9192728890403309 ± 0.03704333576803114 (n=9) | 0.16691174444430443 ± 0.027702951902496296 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| parameter_matched_mlp_vqc | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| random_forest | 9/0/0/0 | 0.9148148148148147 ± 0.017077329199484095 (n=9) | 0.8206710984488762 ± 0.017982665354700004 (n=9) | 0.8055555555555555 ± 0.04098507139310001 (n=9) | 0.874583085109401 ± 0.01186789677138356 (n=9) | 0.6442753111110101 ± 0.05731438384806726 (n=9) | 4464, 4490, 4500, 4502, 4544, 4568, 4980, 5028, 5076 |
| rbf_svm | 9/0/0/0 | 0.873015873015873 ± 0.07155742346261293 (n=9) | 0.792022792022792 ± 0.057648324833362574 (n=9) | 0.6686507936507936 ± 0.05555201235866712 (n=9) | 0.8673054842726104 ± 0.04227041744498207 (n=9) | 0.00343487777798954 ± 0.001081774428782131 (n=9) | NOT MEASURED |

### diabetes — random folds — setting A

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.7438150568869523 ± 0.09409344373740941 (n=9) | 0.7127255460588794 ± 0.03481296128162375 (n=9) | 0.6995642701525054 ± 0.011552876911316049 (n=9) | 0.6143790849673203 ± 0.039215686274509776 (n=9) | 0.006220388888929544 ± 0.0028915356344530385 (n=9) | 5 |
| mlp_default | 9/0/0/0 | 0.6997014443637537 ± 0.09415404501168506 (n=9) | 0.6839189616967395 ± 0.052433672424876 (n=9) | 0.6462599854756718 ± 0.07562325778183511 (n=9) | 0.5307486631016043 ± 0.1242520122503335 (n=9) | 0.16395708888881522 ± 0.009615484028301048 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 9/0/0/0 | 0.4855805696764302 ± 0.23681599130722028 (n=9) | 0.4499841722063944 ± 0.15092900891558464 (n=9) | 0.5 ± 0.0 (n=9) | 0.3455169455169455 ± 0.2595453012043888 (n=9) | 2.2375194777780254 ± 0.19446560372591334 (n=9) | 15 |
| parameter_matched_mlp_vqc | 9/0/0/0 | 0.4274146695715323 ± 0.12154322728725123 (n=9) | 0.6500474833808167 ± 0.016288599182377857 (n=9) | 0.5 ± 0.0 (n=9) | 0.0 ± 0.0 (n=9) | 2.77438143333307 ± 1.5536141526009144 (n=9) | 12 |
| qsvm | 9/0/0/0 | 0.6701767126603728 ± 0.09523564072269543 (n=9) | 0.6628679962013295 ± 0.027289521453943026 (n=9) | 0.5626361655773421 ± 0.057556015356081826 (n=9) | 0.2976608187134503 ± 0.17209757301702502 (n=9) | 8.194211411111407 ± 3.209194608373654 (n=9) | 0 |
| random_forest | 9/0/0/0 | 0.736084886629549 ± 0.08278594324364696 (n=9) | 0.6788540677429566 ± 0.030386581365980308 (n=9) | 0.676325344952796 ± 0.0329523397961528 (n=9) | 0.5891690009337068 ± 0.06601602680666067 (n=9) | 0.6264057777775128 ± 0.07411411409157323 (n=9) | 7168, 7218, 7352, 7540, 7600, 7604, 7682, 7722, 7812 |
| rbf_svm | 9/0/0/0 | 0.7688574195110143 ± 0.12424886348453809 (n=9) | 0.704653371320038 ± 0.09728155834140005 (n=9) | 0.6375090777051561 ± 0.12305253616601672 (n=9) | 0.48540500893442073 ± 0.1886028135357079 (n=9) | 0.0030107555554018794 ± 0.0008791811024848658 (n=9) | NOT MEASURED |
| reupload | 9/0/0/0 | 0.6698297425966271 ± 0.08756894078724121 (n=9) | 0.6418170307059196 ± 0.08859755037685932 (n=9) | 0.6055918663761801 ± 0.1182469776358479 (n=9) | 0.4772114853384203 ± 0.18123249935998956 (n=9) | 63.721494311111705 ± 4.816513595083818 (n=9) | 16 |
| vqc | 9/0/0/0 | 0.6550552731380618 ± 0.09598140562386949 (n=9) | 0.6327951883507439 ± 0.08954206156895675 (n=9) | 0.5993100944081337 ± 0.09627462134931744 (n=9) | 0.4810634354752002 ± 0.12932897264345675 (n=9) | 49.933136444445054 ± 4.43925808656789 (n=9) | 12 |

### diabetes — random folds — setting B

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.7863229242314209 ± 0.08767374527488511 (n=9) | 0.7625830959164293 ± 0.016794624106198884 (n=9) | 0.723202614379085 ± 0.04436838695934112 (n=9) | 0.6289788642729819 ± 0.07407873738182101 (n=9) | 0.005935477777610585 ± 0.001105466206623682 (n=9) | 9 |
| mlp_default | 9/0/0/0 | 0.7222625675784717 ± 0.05713140537403273 (n=9) | 0.6832858499525166 ± 0.038355874694690185 (n=9) | 0.6403413217138707 ± 0.05192354667017702 (n=9) | 0.5182737126471142 ± 0.10196468698530416 (n=9) | 0.1653470777776723 ± 0.011988515162608319 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| parameter_matched_mlp_vqc | 9/0/0/0 | 0.5485475671750182 ± 0.11376378285399716 (n=9) | 0.5830959164292497 ± 0.08962197249419573 (n=9) | 0.5159767610748003 ± 0.0497144233756228 (n=9) | 0.19916068901576148 ± 0.2601372644337272 (n=9) | 3.0173742000010355 ± 1.8301031156668501 (n=9) | 11 |
| random_forest | 9/0/0/0 | 0.7514968127168562 ± 0.06120709268382999 (n=9) | 0.7420069642291864 ± 0.04858178751159988 (n=9) | 0.7132171387073348 ± 0.06931786062320039 (n=9) | 0.6216907638476266 ± 0.10339560010626274 (n=9) | 0.61942006666666 ± 0.0656310780493089 (n=9) | 7436, 7468, 7572, 8044, 8056, 8162, 8180, 8206, 8264 |
| rbf_svm | 9/0/0/0 | 0.7977729363350278 ± 0.08009969507009736 (n=9) | 0.7171573282684394 ± 0.0749527083874224 (n=9) | 0.6497095134350036 ± 0.08555282249157943 (n=9) | 0.4930994930994931 ± 0.1514447334326672 (n=9) | 0.003228222222080351 ± 0.0012912273307722707 (n=9) | NOT MEASURED |

### Parkinson's — grouped folds — setting A

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.880952380952381 ± 0.05261042808091512 (n=9) | 0.7640075973409306 ± 0.09597429685564049 (n=9) | 0.7138888888888889 ± 0.13910371707581506 (n=9) | 0.8315980315980316 ± 0.08533138424356536 (n=9) | 0.007814066666671553 ± 0.0032800574457695175 (n=9) | 5 |
| mlp_default | 9/0/0/0 | 0.8117724867724867 ± 0.1265107744996444 (n=9) | 0.8179803735359291 ± 0.09031209212540475 (n=9) | 0.7345238095238095 ± 0.18587447303998242 (n=9) | 0.881979591451815 ± 0.05710976048387276 (n=9) | 0.22107242222237902 ± 0.06716349464722969 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 9/0/0/0 | 0.6265873015873016 ± 0.16492607256110492 (n=9) | 0.5834124723013612 ± 0.2506425447717701 (n=9) | 0.5 ± 0.0 (n=9) | 0.5714873059923938 ± 0.4286820260822748 (n=9) | 4.240375977778058 ± 4.269016690436787 (n=9) | 15 |
| parameter_matched_mlp_vqc | 9/0/0/0 | 0.4932539682539682 ± 0.14080212138537299 (n=9) | 0.7502374169040835 ± 0.014245014245014287 (n=9) | 0.5 ± 0.0 (n=9) | 0.8572309589885908 ± 0.009250693802035137 (n=9) | 3.0463810333331924 ± 0.8176953177648922 (n=9) | 12 |
| qsvm | 9/0/0/0 | 0.7464285714285714 ± 0.13671700375464005 (n=9) | 0.7959797404241848 ± 0.04494961019322829 (n=9) | 0.6623015873015873 ± 0.15046460549135443 (n=9) | 0.8731880921260853 ± 0.018705762318128715 (n=9) | 7.412456611111136 ± 0.88658888579211 (n=9) | 0 |
| random_forest | 9/0/0/0 | 0.7067460317460317 ± 0.2754061883729273 (n=9) | 0.8385565052231719 ± 0.06538756363264106 (n=9) | 0.7638888888888888 ± 0.14690809018379908 (n=9) | 0.8948100274187231 ± 0.041442488465430374 (n=9) | 0.8784105888888816 ± 0.239039709263122 (n=9) | 4254, 4268, 4446, 4804, 4858, 4928, 5526, 5638, 5722 |
| rbf_svm | 9/0/0/0 | 0.6857142857142857 ± 0.3243321238030736 (n=9) | 0.8179803735359291 ± 0.09031209212540475 (n=9) | 0.7345238095238095 ± 0.18587447303998242 (n=9) | 0.881979591451815 ± 0.05710976048387276 (n=9) | 0.009288077777758494 ± 0.018831647191741675 (n=9) | NOT MEASURED |
| reupload | 9/0/0/0 | 0.6891534391534392 ± 0.28123752456864215 (n=9) | 0.7098765432098766 ± 0.14357462159778908 (n=9) | 0.6288359788359789 ± 0.23427499021031856 (n=9) | 0.8077239030349259 ± 0.09062796218618642 (n=9) | 59.440486333333155 ± 9.152799570103092 (n=9) | 16 |
| vqc | 9/0/0/0 | 0.6404761904761904 ± 0.2686199455924076 (n=9) | 0.6546375435264324 ± 0.16512409684164378 (n=9) | 0.5835978835978836 ± 0.205814516354783 (n=9) | 0.7511655885283872 ± 0.14423168396968516 (n=9) | 43.11949024444463 ± 4.5672120110771175 (n=9) | 12 |

### Parkinson's — grouped folds — setting B

Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.

| Model | Status counts | AUC mean ± SD (n) | Accuracy mean ± SD (n) | Balanced accuracy mean ± SD (n) | F1 mean ± SD (n) | Fit seconds mean ± SD (n) | Trainable parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 9/0/0/0 | 0.7805555555555556 ± 0.12472901465779992 (n=9) | 0.7141500474833808 ± 0.127403317315305 (n=9) | 0.6496031746031746 ± 0.1441803795924065 (n=9) | 0.7954133700263731 ± 0.11337703112411408 (n=9) | 0.011728533333452005 ± 0.013066577380400003 (n=9) | 23 |
| mlp_default | 9/0/0/0 | 0.7707671957671958 ± 0.1423118450786338 (n=9) | 0.7220639442861665 ± 0.1025627286144827 (n=9) | 0.6448412698412698 ± 0.13947382494864402 (n=9) | 0.8059224784381803 ± 0.09104415909838887 (n=9) | 0.15882285555603126 ± 0.01702146130557119 (n=9) | NOT MEASURED |
| parameter_matched_mlp_reupload | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| parameter_matched_mlp_vqc | 0/0/0/9 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| random_forest | 9/0/0/0 | 0.8958994708994709 ± 0.05380607049835713 (n=9) | 0.7562519784742007 ± 0.14352239686467286 (n=9) | 0.698015873015873 ± 0.16127657368618245 (n=9) | 0.8174640831037238 ± 0.135431938234487 (n=9) | 0.6976061333334656 ± 0.06396075921950552 (n=9) | 3804, 3828, 3860, 4134, 4140, 4174, 5208, 5222, 5366 |
| rbf_svm | 9/0/0/0 | 0.667063492063492 ± 0.239827773852636 (n=9) | 0.7388414055080722 ± 0.10135004789912112 (n=9) | 0.6054232804232804 ± 0.13265280079242608 (n=9) | 0.8271316975757309 ± 0.08489692818775933 (n=9) | 0.003860033333290226 ± 0.002067218743012757 (n=9) | NOT MEASURED |

## Fold-paired comparisons

The primary difference is best quantum AUC minus best classical AUC on the same dataset, split, seed, and fold, using feature setting A. The best model can change by fold, so this is a best-of-fold comparison, not a fixed model-pair estimate. The paired tests use nine correlated pairs: the same three folds are reused across three estimator seeds. Therefore, paired t-test and Wilcoxon p-values are exploratory and are not independent-sample evidence; p-values are also unadjusted for multiple comparisons.

| Dataset / split | Mean difference | Sample SD | n | Paired t-test p | Wilcoxon p | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| breast_cancer / random | -0.04325980392156863 | 0.039767534515865934 | 9 | 0.0114679412790751 | 0.00390625 | quantum lower; both tests p < 0.05 |
| heart_disease / random | -0.12470831637498306 | 0.08161277522869564 | 9 | 0.001792246854924303 | 0.00390625 | quantum lower; both tests p < 0.05 |
| parkinsons / random | -0.15687830687830692 | 0.06948949785702335 | 9 | 0.00014171323118845961 | 0.00390625 | quantum lower; both tests p < 0.05 |
| parkinsons / grouped | -0.096957671957672 | 0.13089164055487038 | 9 | 0.05698239188796903 | 0.0390625 | tests disagree (t-test >= 0.05; Wilcoxon < 0.05) |
| diabetes / random | -0.07933914306463329 | 0.03998317666775984 | 9 | 0.0003408535030197856 | 0.0078125 | quantum lower; both tests p < 0.05 |

Decision wording uses the requested p-value threshold of 0.05: the breast-cancer, heart-disease, random Parkinson, and diabetes mean differences favor classical (negative quantum-minus-classical differences) with both tests below 0.05; Parkinson grouped tests disagree (t-test p >= 0.05; Wilcoxon p < 0.05). These small-sample tests are exploratory.

### ParameterMatchedMLP versus each quantum model

AUC differences are ParameterMatchedMLP minus the named quantum model on matching setting A folds. Each entry gives mean difference, sample SD, n, paired t-test p, and Wilcoxon p. Architecture counts on setting A were 12 for VQC-matched and 15 for Reupload-matched. Grouped Parkinson's uses the subject-grouped folds.

#### breast cancer — random

| ParameterMatchedMLP | Quantum model | Mean difference | Sample SD | n | t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| parameter_matched_mlp_vqc | qsvm | -0.16262254901960785 | 0.2267629206889752 | 9 | 0.0636320209250788 | 0.23828125 |
| parameter_matched_mlp_vqc | vqc | 0.05290032679738564 | 0.3492268480084339 | 9 | 0.6615921786353989 | 0.8203125 |
| parameter_matched_mlp_vqc | reupload | -0.024509803921568638 | 0.2274315341480013 | 9 | 0.754754600052047 | 0.8203125 |
| parameter_matched_mlp_reupload | qsvm | -0.1423611111111111 | 0.22848508826538444 | 9 | 0.09852872588599243 | 0.41015625 |
| parameter_matched_mlp_reupload | vqc | 0.07316176470588238 | 0.3265230089240786 | 9 | 0.5204002114478247 | 0.5703125 |
| parameter_matched_mlp_reupload | reupload | -0.0042483660130718925 | 0.23129552114905724 | 9 | 0.9574076248147254 | 1.0 |

#### heart disease — random

| ParameterMatchedMLP | Quantum model | Mean difference | Sample SD | n | t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| parameter_matched_mlp_vqc | qsvm | -0.06193359110025776 | 0.1887076107809167 | 9 | 0.35365967280043265 | 0.359375 |
| parameter_matched_mlp_vqc | vqc | -0.011936982770316086 | 0.16261273830212947 | 9 | 0.8312139447269266 | 0.65234375 |
| parameter_matched_mlp_vqc | reupload | 0.031225410392077053 | 0.17566403833745084 | 9 | 0.608339177436416 | 0.734375 |
| parameter_matched_mlp_reupload | qsvm | -0.056534052367385686 | 0.1437988199787786 | 9 | 0.2721062193010978 | 0.49609375 |
| parameter_matched_mlp_reupload | vqc | -0.006537444037444014 | 0.14780310757706563 | 9 | 0.8977136186460837 | 1.0 |
| parameter_matched_mlp_reupload | reupload | 0.036624949124949124 | 0.10459152769091587 | 9 | 0.32417721287622225 | 0.25 |

#### Parkinson's — random

| ParameterMatchedMLP | Quantum model | Mean difference | Sample SD | n | t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| parameter_matched_mlp_vqc | qsvm | -0.14199735449735446 | 0.18091838435516172 | 9 | 0.04634660212406741 | 0.0703125 |
| parameter_matched_mlp_vqc | vqc | 0.004828042328042311 | 0.17925623760790135 | 9 | 0.9375848727432077 | 1.0 |
| parameter_matched_mlp_vqc | reupload | -0.0685846560846561 | 0.1699886246990807 | 9 | 0.26067118700295844 | 0.30078125 |
| parameter_matched_mlp_reupload | qsvm | -0.1340608465608465 | 0.23348432951754922 | 9 | 0.12327256208330588 | 0.1875 |
| parameter_matched_mlp_reupload | vqc | 0.01276455026455028 | 0.2739500660128235 | 9 | 0.8922863295198997 | 0.96875 |
| parameter_matched_mlp_reupload | reupload | -0.060648148148148125 | 0.17239939957273945 | 9 | 0.32208462966514834 | 0.359375 |

#### Parkinson's — grouped

| ParameterMatchedMLP | Quantum model | Mean difference | Sample SD | n | t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| parameter_matched_mlp_vqc | qsvm | -0.25317460317460316 | 0.15061729322781756 | 9 | 0.0009982184276098908 | 0.00390625 |
| parameter_matched_mlp_vqc | vqc | -0.1472222222222222 | 0.3041024475585672 | 9 | 0.18446627029837073 | 0.09765625 |
| parameter_matched_mlp_vqc | reupload | -0.1958994708994709 | 0.2442245737031364 | 9 | 0.0427486446514936 | 0.0546875 |
| parameter_matched_mlp_reupload | qsvm | -0.11984126984126982 | 0.2392020405706271 | 9 | 0.1712408482317672 | 0.1953125 |
| parameter_matched_mlp_reupload | vqc | -0.013888888888888852 | 0.36107851326790985 | 9 | 0.9109759235479428 | 0.91015625 |
| parameter_matched_mlp_reupload | reupload | -0.06256613756613755 | 0.35448720544383716 | 9 | 0.6108378011788957 | 0.5703125 |

#### diabetes — random

| ParameterMatchedMLP | Quantum model | Mean difference | Sample SD | n | t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| parameter_matched_mlp_vqc | qsvm | -0.24276204308884047 | 0.16208312284905918 | 9 | 0.0020199438789196564 | 0.0078125 |
| parameter_matched_mlp_vqc | vqc | -0.2276406035665295 | 0.17585742770168783 | 9 | 0.00465249405124849 | 0.01171875 |
| parameter_matched_mlp_vqc | reupload | -0.24241507302509477 | 0.14521696435082374 | 9 | 0.0010423732665590126 | 0.00390625 |
| parameter_matched_mlp_reupload | qsvm | -0.18459614298394256 | 0.23574168567266623 | 9 | 0.0467445234973095 | 0.07421875 |
| parameter_matched_mlp_reupload | vqc | -0.16947470346163157 | 0.2585405630806329 | 9 | 0.08479938742024737 | 0.12890625 |
| parameter_matched_mlp_reupload | reupload | -0.18424917292019685 | 0.22703069226589426 | 9 | 0.04090241242499053 | 0.0546875 |

### ParameterMatchedMLP setting B eligibility

Setting B parameter-matched architectures were measured only where the architecture audit found an eligible real-class trainable-parameter count within 10% of a quantum target. Heart disease / Reupload target: hidden_dims=[1], 16 parameters. Diabetes / VQC target: hidden_dims=[1], 11 parameters. Their fold metrics are included in the setting B tables. Other setting B ParameterMatchedMLP variants are NOT MEASURED because no eligible architecture was identified by the real class parameter audit: breast cancer (minimum candidate 31 parameters), Parkinson's (minimum 23), heart disease VQC match (14 is outside the ±10% target interval around 12), and diabetes Reupload match (no candidate in range). No quantum runs were made for setting B.

## Protocol, commands, and validation

### Data, splits, and seeds

The datasets are the scikit-learn breast cancer data and the repository files `data/raw/heart_cleveland.data`, `data/raw/parkinsons.data`, and `data/raw/pima_diabetes.csv`. Dataset sample counts, feature counts, label counts, fold indices, source row indices, and Parkinson subject IDs are saved in `splits.json`. The current sample has 80 rows per dataset, selected directly from the full dataset with sample seed 42. Random-fold membership uses `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`. Fit seeds are 42, 1729, and 2024; the same split indices are used across models and fit seeds. Fold sizes (train/test) are 53/27, 53/27, and 54/26.

Parkinson's data contained 29 represented subjects in the 80-row sample. The recorded per-subject row counts are: `{"S01":1,"S02":1,"S04":3,"S05":4,"S06":3,"S07":2,"S08":3,"S10":3,"S13":4,"S16":3,"S17":1,"S18":2,"S19":3,"S20":5,"S21":4,"S22":3,"S24":4,"S27":3,"S31":3,"S32":4,"S33":2,"S34":1,"S35":4,"S37":3,"S39":1,"S42":4,"S43":1,"S49":3,"S50":2}`. Random-fold train/test subject overlaps were 15, 16, and 15 for folds 0, 1, and 2. The grouped sample uses the same direct 80-row draw, including repeated subject recordings because one row per subject cannot provide 80 rows; its grouped fold overlaps were 0, 0, and 0. Grouped train/test subject counts by fold were 19/10, 19/10, and 20/9. Thus Parkinson's random-fold results are subject-leaked; use the grouped results to assess subject separation.

### Models and implementation

Classical models: LogisticRegressionModel, RandomForestModel, RBFSVMModel, MLPClassifier (`early_stopping=False`, `max_iter=500`), plus the repository ParameterMatchedMLP. Quantum models: QSVM with `zz` feature map, VQC with angle/RX encoding and one layer, and ReuploadClassifier with one layer and repeated Rot plus RY(x+phi) encoding and ring CNOT entanglement. All use PennyLane `default.qubit`, 4 qubits, and `shots=None`; no real quantum hardware ran. QSVM has 0 trainable parameters; VQC and Reupload have 12 and 16. QSVM `circuit_depth=3` is feature-map circuit depth metadata, not trained variational layers. HybridQNN was not run because source inspection shows its fit delegates to a random-forest mock.

### Leakage, timeouts, and timing

The fold-local audit passed: preprocessing is fit on training rows only, test rows receive transform only, and the estimator fits training labels only. Grouped Parkinson folds had zero subject overlap. Timeout self-test passed: child sleep 30 seconds; parent timeout 5 seconds; Windows taskkill /T /F; measured elapsed 5.896005899999977 seconds. Each grid fit used a child process and 900-second parent timeout. Classical benchmark statuses after deduplication: 468 completed, 0 timeout, 0 error, 72 not measured. Quantum benchmark statuses after deduplication and corrected fold-0 replacements: 135 completed, 0 timeout, 0 error, 0 not measured. Earlier timing pilots and repeats are excluded from score counts.

Heart timing gate, seed 42, fold 0, n=80, train/test 53/27: QSVM fit 6.996467499999881 s, predict 7.0287773999998535 s; 2862 kernel evaluations combined across fit and predict, not split. VQC fit 55.2009594000001 s, predict 0.132425000000012 s. Reupload fit 75.10359410000001 s, predict 0.14134219999982633 s.

Repeated same-fold heart QSVM estimator fit times read from `timing_repeats_heart_qsvm.json` only (seed 42, n=80, train/test 53/27): [9.510194099999353, 7.171762999998464, 8.194112699999096, 8.036131099997874, 8.948771900002612]. Measured min=7.171762999998464, median=8.194112699999096, max=9.510194099999353. Preprocessing is excluded from estimator train_seconds; end-to-end pipeline time is NOT MEASURED.

### Sanity check and earlier result mismatch

The requested direct-sample breast-cancer logistic-regression seed-42 fold-0 sanity check did not match the earlier record: current direct sample AUC=0.976470588235294, accuracy=0.9259259259259259, balanced accuracy=0.9411764705882353; earlier record AUC=0.9941176470588236, accuracy=0.9629629629629629, balanced accuracy=0.9705882352941176. Fold sizes matched at train/test 53/27. Investigation found that the earlier shared-80 artifact selected 80 rows from a prior 150-row sample; a separate two-stage diagnostic reproduced that earlier record exactly (AUC=0.9941176470588236, accuracy=0.9629629629629629, balanced accuracy=0.9705882352941176). This benchmark follows the task's final instruction to draw 80 rows directly from each full dataset. The mismatch is therefore a sample-protocol difference; old scores were not carried into the new results.

### Reproduction commands and evidence files

Run commands from the repository root in PowerShell. The exact command line, raw stdout/stderr, and exit status are saved as the adjacent `.command.txt`, `.raw.txt`, and `.exit.txt` evidence files where applicable.

| Stage | Command / evidence |
| --- | --- |
| Environment and repository inventory | `step1_package_versions.command.txt` / `step1_package_versions.raw.txt`; `step1_repo_inventory.*` |
| Split construction | `python results-v2/make_splits.py`; `make_splits.*`, `final_splits.*`; manifest `splits.json` |
| Child timeout test | `timeout_selftest.command.txt`; `timeout_selftest.json`, `.raw.txt`, `.exit.txt` |
| Classical benchmark grid | `classical_grid.command.txt`; `classical_grid.raw.txt`, `.exit.txt`; fold records `classical_raw.json` |
| Quantum benchmark grid | `quantum_grid.command.txt`; `quantum_grid.raw.txt`, `.exit.txt`; fold records `quantum_raw.json` |
| Leakage and model audits | `audit_fold_leakage.*`, `audit_model_implementations.*`, `audit_pm_architecture.*`, `audit_qsvm_parameter_count.*` |
| Timing gate / repeat | `timing_gate_direct_heart_{qsvm,vqc,reupload}.*`; `repeat_qsvm_timing.*` |
| Summaries and CSV validation | `summarize_results.command.txt`; `summarize_results.raw.txt`, `.exit.txt`; `print_derived_summaries.*` |
| Per-fold result evidence | `classical_raw.json`, `quantum_raw.json`, `fold_logs/`, `summary.csv`, `paired_differences.csv` |

Package versions observed in the saved `pip show` output: scikit-learn 1.9.1, PennyLane 0.45.1, NumPy 2.5.3. The CSV validator printed three sample data rows from each CSV, checked every CSV header against the required columns, and checked row content against source JSON records; see `summarize_results.raw.txt` and `analysis_summary.json` `csv_audit`.

## Limitations

- This is a single small 80-row sample per dataset, not a full-data performance estimate. The split seed is fixed; fold uncertainty does not capture variation across independent subsamples.
- Each model used three folds for each of three estimator seeds; the folds are reused across seeds, so the nine paired observations are not nine independent datasets.
- Parkinson's random-fold evaluations leak subjects across train/test. The grouped variant removes subject overlap but still has repeated recordings in the 80-row sample and has only three grouped folds.
- Classical setting A is restricted to four PCA features to match the quantum input dimension; this constrains classical baselines. Setting B is classical only and does not support a direct quantum comparison.
- Paired best-of-fold comparisons select a different best model when needed in each fold, and p-values are exploratory, unadjusted, and based on a small number of pairs.
- Quantum results are analytic simulator results only (default.qubit); no hardware noise, shot noise, queue delay, or hardware execution was measured.
- Quantum calibration (Brier score/ECE), conformal prediction, SHAP, nested cross-validation, and full-data quantum runs are NOT MEASURED. Real hardware results are NOT MEASURED because only a simulator was configured.
- MLP early stopping was disabled in this benchmark. This does not establish that the earlier low-AUC behavior was caused solely by early stopping; the earlier 150-row separate rerun is not this benchmark.
- Estimator fit time excludes preprocessing and process startup. End-to-end user-facing latency is NOT MEASURED.
- The earlier sanity comparison differs because the older sample protocol was two-stage; the new benchmark uses direct-from-full-data sampling as instructed.

## Safe to claim

- Benchmarked four classical baselines and three PennyLane quantum classifiers on 80-row stratified samples from four biomedical datasets, with three-fold CV and fold-local preprocessing; Parkinson's was also evaluated with subject-grouped folds.
- The quantum models ran on PennyLane's analytic `default.qubit` simulator with four qubits and four PCA features; no real hardware was used.
- Implemented parent-enforced subprocess timeouts and fold-level result checkpointing; the saved timeout self-test passed.
- The grouped Parkinson's split had zero subject overlap, while random-fold subject overlap was observed and reported.

## NOT safe to claim

- Do not claim quantum advantage or that quantum models beat classical models generally; the benchmark is a small fixed sample with best-of-fold selection and exploratory tests.
- Do not present Parkinson's random-fold results as subject-independent generalization.
- Do not claim full-dataset quantum performance, real-hardware performance, calibration, conformal coverage, SHAP explanations, or nested cross-validation.
- Do not claim full-feature classical comparisons against quantum models; setting A classical baselines use four PCA features, and quantum models were not run in setting B.
- Do not claim the older shared-80 results were reproduced by the direct 80-row sample; they match only under the separate two-stage sampling diagnostic.


## Audit corrections (raw-record review)

The raw quantum audit found one error record: `heart_disease / random / qsvm / seed 42 / fold 0 / setting A`, with no AUC. The exception was `IndexError: index 301 is out of bounds for axis 0 with size 297`. `heart_cleveland.data` contains 303 source rows; dropping rows with missing values leaves 297 model rows, while the largest retained original source index is 301. Source row IDs are labels for auditing and can have gaps after cleaning; indexing the 297-row cleaned array by those IDs raises the error. The fix in `run_fold.py` uses the manifest's `sample_local_row_indices` for array indexing and validates the positions. The sample and split were not changed.

`quantum_raw.json` contains 146 records and three duplicate keys (11 additional raw rows across those keys). The collisions are heart QSVM, VQC, and Reupload at seed 42, fold 0, setting A; they include non-benchmark timing pilots and QSVM repeat measurements. The audit and every excluded collision with its AUC are in `audit_raw_records.raw.txt` and `dedup_rule.md`. Score CSVs are rebuilt from benchmark-grid rows only, keeping the first completed record for each requested key. Corrected reruns for all three conflicting model keys are in `quantum_rerun_heart_fold0.json` and replace the corresponding benchmark rows.

The corrected fold AUCs did not change versus the retained benchmark values: QSVM `0.5166666666666666`, VQC `0.7111111111111111`, and Reupload `0.4166666666666667`. Earlier two-stage pilot AUCs differ for VQC (`0.6166666666666667`) and Reupload (`0.48333333333333334`); those came from a different sample protocol and are excluded from benchmark summaries.

Status counts above describe the deduplicated benchmark set, not every raw record or the NOT MEASURED placeholder rows in the CSV. The raw error remains documented and is not counted as a benchmark-grid error after the corrected fold rerun. `summary.csv` and `paired_differences.csv` were rebuilt; their required columns and first three data rows were printed and checked in `deduplicate_rebuild.raw.txt`.
