# Adsorption Uptake Prediction
## Files

```text
checkpoint_best.pt
export_predictions.py
run_export.sh
requirements.txt
data/
demo_cifs/
models/
tasks/
tools/
```

Example data files:

```text
data/demo_prediction_input.xlsx
data/descriptors_example.xlsx
data/mof_local_chemical_descriptors_example.csv
```

## Install

```bash
pip install -r requirements.txt
```

## Run

To run the included small demo:

```bash
bash run_export.sh demo_cifs
```

For new MOFs, place CIF files in one folder using the naming format `<coreid>.cif`, where `coreid` matches the IDs in the input and descriptor tables.

The default checkpoint is:

```text
checkpoint_best.pt
```

The output folder is:

```text
prediction_outputs/
```

CO2 is component 0. The competing gas X is component 1.
