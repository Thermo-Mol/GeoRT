# Molecule files

Place refrigerant molecular structure files in this folder.

Supported formats are handled by RDKit and include `.mol`, `.sdf`, and related molecule file formats used by the parser.

The default configuration uses the following mapping when file names differ from refrigerant names:

```yaml
molecule_file_map:
  R236fa: ER236FA.mol
  R1234zeE: R1234zee.mol
  R1366mzz(Z): R1366Z.mol
  R1225ye(Z): R1225ye(z).mol
```
