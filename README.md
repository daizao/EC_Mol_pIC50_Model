# EC_Mol_pIC50_Model
```
EC_Mol_pIC50_Model is a deep learning-based model for predicting the pIC50 values of small-molecule compounds in esophageal cancer.
```

## Dependence

```
Pytorch
Pandas
Numpy
Sklearn
Transformers
```

## Sample

```
python smile_predict.py --help
    usage: smile_predict.py [-h] --input INPUT [--outdir OUTDIR] [--outfile OUTFILE]

    Calculate the IC50 of drugs (Smile format) based on the esophageal cancer drug model

    options:
      -h, --help            show this help message and exit
      --input INPUT, -i INPUT
                            Drug file (Smile format)
      --outdir OUTDIR       The output directory to which all files should be written
      --outfile OUTFILE, -o OUTFILE
                            The output file name

```

