# GPCR Agonist Prediction with ESM-2

CS/DS 541 final project. 
Projectmates:
Nathan Fass, 
Marc Godbout-Chouinard, 
Andrey Sargsyan,
Joseph DiPietro

## Setup


pip install -r requirements.txt
python -m src.data.dataset

Use to run data. Once loaded, it is cached, and shouldnt be loaded again.

## Files

All of this lands in `data/`.

`labels.csv` - raw label table 

`receptors.fasta` - 65 receptor sequences 

`peptides.fasta` - 339 peptide sequences 

`pairs.csv` - labels joined to sequences 

`pdb/` - the 403 downloaded PDB files 

## Dataset

`Pairs` - 22035

`Receptors` - 65

`Peptides` - 339

`Agonists` - 457, or 2.07 percent

`Imbalance` - 47.2 to 1

`Receptor length` - 321 to 536, mean 416

`Peptide length` - 3 to 37, mean 12

Every receptor was tested against every peptide. 65 x 339 = 22035.

Most agonists sit on 7 receptors. DMSR-7-1 alone has 61. Watch this when
splitting, or one fold takes nearly all the positives.

## Why the download is small

The HuggingFace repo is 77.7 GB. Most of it is AlphaFold output we skip.

`pair_representations` - 64.3 GB, skipped

`relaxed_PDBs` - 11.5 GB, we take 403 of 22035 files

`DeorphaNN_training` - 1.8 GB, we take the label CSV only

`DeorphaNN_scores` - 966 KB, not yet, needed later for the baseline

The repo has no FASTA file. Sequences only live inside the PDB files, receptor
as chain A and peptide as chain B.

Since every receptor was tested against every peptide, one receptor folder
holds all 339 peptides. One file from each other receptor gives the remaining
64 receptors. So we need 403 files, not 22035.

## Layout

```
src/data/download.py   fetch the label CSV and the PDB subset
src/data/dataset.py    parse PDBs, join sequences, write pairs.csv
```

## Status

Done: raw data import.

Not done: EDA, preprocessing and splits, ESM-2 embeddings, model.

This code loads sequences, not embeddings. Turning the 404 sequences into
per-residue tensors is a separate GPU step.
