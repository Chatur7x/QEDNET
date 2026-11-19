import inspect

from qednet.models.quantum.qsvm import QSVM
from qednet.models.quantum.vqc import VQC
from qednet.models.quantum.reupload import ReuploadClassifier
from qednet.models.quantum.hybrid_qnn import HybridQNN
from qednet.models.quantum.mocks import MockHybridQNN
from qednet.encoding import zz_feature_map

for label, obj in (
    ("QSVM._get_encoding_fn", QSVM._get_encoding_fn),
    ("zz_feature_map", zz_feature_map),
    ("VQC._build", VQC._build),
    ("ReuploadClassifier._build", ReuploadClassifier._build),
    ("HybridQNN.fit", HybridQNN.fit),
    ("MockHybridQNN.__init__", MockHybridQNN.__init__),
):
    print(f"===== {label} =====")
    print(inspect.getsource(obj))
