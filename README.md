# GeoRT

GeoRT is a geometry aware Transformer model for predicting the liquid thermal conductivity of HFC and HFO refrigerants from molecular structure and thermodynamic state parameters.

The model inputs include atom types, bond types, molecular connectivity, three dimensional interatomic distances, temperature, pressure, critical temperature, critical pressure, reduced temperature, and reduced pressure.

## Repository Structure

```text
configs/        Model configuration files
data/           Thermal conductivity data, R1216 prediction conditions, and molecular structure files
checkpoints/    Clean public GeoRT checkpoint for prediction
scripts/        Dataset checking script
src/            Source code of GeoRT
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

If RDKit cannot be installed by pip, install it with conda:

```bash
conda install -c conda-forge rdkit
```

## Dataset Check

```bash
python scripts/check_dataset.py --config configs/geort_groupdro_monotonic.yaml
```

## Training

```bash
python -m refrig_geotransformer.train --config configs/geort_groupdro_monotonic.yaml
```

## Prediction

```bash
python -m refrig_geotransformer.predict \
  --checkpoint checkpoints/geort_r1216_external.pt \
  --molecule data/mol/R1216.mol \
  --conditions data/r1216_conditions.csv \
  --output outputs/predictions_R1216.csv
```

