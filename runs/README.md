# Local experiment records

The JSON result, certificate, and status records are tracked as experiment outputs. Serialized Python model and preprocessing objects (`*.pkl`) are excluded by the repository-wide ignore rule; do not load pickle files from untrusted sources. Reproducing predictions requires rerunning the experiment to create compatible model artifacts.
