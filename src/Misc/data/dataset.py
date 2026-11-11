"""
Build the GPCR and peptide pair table.

Reads sequences out of the downloaded PDB files, then joins them onto the
label rows. No preprocessing and no splitting happen here.

Run it with:
    python -m src.data.dataset
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import download as download_tools

# PDB three letter residue codes to single letters.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Chain layout is fixed by how DeorphaNN built the files.
RECEPTOR_CHAIN = "A"
PEPTIDE_CHAIN = "B"


def read_chains_from_pdb(pdb_path):
    """Return a dict of chain letter to sequence for one PDB file."""
    seen = set()
    chains = {}

    pdb_file = open(pdb_path, "r")
    for line in pdb_file:
        if not line.startswith("ATOM"):
            continue

        # Column positions are fixed by the PDB spec.
        chain = line[21]
        residue_number = line[22:27]
        residue_name = line[17:20].strip()

        # Many atoms share a residue. Record each residue once.
        key = (chain, residue_number)
        if key in seen:
            continue
        seen.add(key)

        if chain not in chains:
            chains[chain] = []
        chains[chain].append(THREE_TO_ONE.get(residue_name, "X"))
    pdb_file.close()

    result = {}
    for chain in chains:
        result[chain] = "".join(chains[chain])
    return result


def parse_peptide_name(file_name, receptor_name):
    """
    Pull the peptide name out of a PDB file name.

    Names look like amber_r_AEX-2-1_FLP-1-1_unrelaxed_rank_003_...pdb
    Returns None if the name does not fit.
    """
    name = file_name

    prefix = "amber_r_"
    if name.startswith(prefix):
        name = name[len(prefix):]

    receptor_prefix = receptor_name + "_"
    if not name.startswith(receptor_prefix):
        return None
    name = name[len(receptor_prefix):]

    cut = name.find("_unrelaxed")
    if cut == -1:
        return None
    return name[:cut]


def extract_all_sequences(pdb_folder):
    """Read every PDB file. Returns receptor and peptide name to sequence."""
    receptors = {}
    peptides = {}

    for receptor_name in sorted(os.listdir(pdb_folder)):
        receptor_path = os.path.join(pdb_folder, receptor_name)
        if not os.path.isdir(receptor_path):
            continue

        for file_name in sorted(os.listdir(receptor_path)):
            if not file_name.endswith(".pdb"):
                continue

            peptide_name = parse_peptide_name(file_name, receptor_name)
            if peptide_name is None:
                continue

            chains = read_chains_from_pdb(os.path.join(receptor_path, file_name))

            if RECEPTOR_CHAIN in chains:
                if receptor_name not in receptors:
                    receptors[receptor_name] = chains[RECEPTOR_CHAIN]

            if PEPTIDE_CHAIN in chains:
                if peptide_name not in peptides:
                    peptides[peptide_name] = chains[PEPTIDE_CHAIN]

    return receptors, peptides


def write_fasta(sequences, out_path):
    """Write a name to sequence dict as FASTA."""
    out_file = open(out_path, "w")
    for name in sorted(sequences.keys()):
        out_file.write(">" + name + "\n")
        out_file.write(sequences[name] + "\n")
    out_file.close()


def read_fasta(in_path):
    """Read a FASTA file into a name to sequence dict."""
    sequences = {}
    name = None
    parts = []

    in_file = open(in_path, "r")
    for line in in_file:
        line = line.strip()
        if line == "":
            continue

        if line.startswith(">"):
            if name is not None:
                sequences[name] = "".join(parts)
            name = line[1:]
            parts = []
        else:
            parts.append(line)
    in_file.close()

    if name is not None:
        sequences[name] = "".join(parts)
    return sequences


def build_pair_table(data_folder, receptors, peptides):
    """Join sequences onto the label rows. Returns a DataFrame."""
    table = pd.read_csv(os.path.join(data_folder, "labels.csv"))
    table = table.rename(columns={
        "GPCR name": "receptor",
        "Peptide": "peptide",
        "binds": "label",
        "pair": "pair",
    })
    table["label"] = table["label"].astype(int)

    receptor_column = []
    peptide_column = []
    for row_number in range(len(table)):
        receptor_name = table["receptor"].iloc[row_number]
        peptide_name = table["peptide"].iloc[row_number]
        receptor_column.append(receptors.get(receptor_name, ""))
        peptide_column.append(peptides.get(peptide_name, ""))

    table["receptor_seq"] = receptor_column
    table["peptide_seq"] = peptide_column
    return table


class GpcrPairDataset(Dataset):
    """
    Pairs of receptor and peptide with a binary label.

    Items hold names, sequences, and the label. Embeddings are not built here.
    That needs ESM-2 and a GPU.
    """

    def __init__(self, table, row_indices=None):
        if row_indices is None:
            self.table = table.reset_index(drop=True)
        else:
            self.table = table.iloc[row_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        return {
            "receptor": row["receptor"],
            "peptide": row["peptide"],
            "receptor_seq": row["receptor_seq"],
            "peptide_seq": row["peptide_seq"],
            "label": torch.tensor(float(row["label"])),
        }


def main():
    """Download, read sequences, write FASTA and pairs.csv."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(project_root, "data")

    pdb_folder = download_tools.download_dataset(data_folder)

    receptors, peptides = extract_all_sequences(pdb_folder)
    print("receptors: " + str(len(receptors)))
    print("peptides:  " + str(len(peptides)))

    write_fasta(receptors, os.path.join(data_folder, "receptors.fasta"))
    write_fasta(peptides, os.path.join(data_folder, "peptides.fasta"))

    table = build_pair_table(data_folder, receptors, peptides)
    table.to_csv(os.path.join(data_folder, "pairs.csv"), index=False)
    print("rows: " + str(len(table)))


if __name__ == "__main__":
    main()
