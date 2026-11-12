# Makes ESM-2 embeddings for the C. elegans sequences.
# Reads the two sequence CSVs from the paper, writes one npz per model and kind.
# Run it once. Training just loads the cache after that.
# Nothing here edits the CSVs. Blank padding rows get skipped on read.
#
# Run it with:
#     python -m src.embed

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# 650M is for the main model, 150M is for the model size ablation
MODELS = {
    "650M": "facebook/esm2_t33_650M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
}

# Sequences per forward pass
BATCH_SIZE = 8

# CSV identifiers
SOURCES = {
    "receptors": ("C. Elegans GPCR.csv", "GPCR name", "GPCR sequence"),
    "peptides": ("C. Elegans Peptides.csv", "Peptide name", "Peptide sequence"),
}


# Root dir of repo
def project_root():
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(here))


# Pulls a name to sequence dict out of one CSV
def read_sequence_csv(csv_path, name_column, sequence_column):
    table = pd.read_csv(csv_path)
    sequences = {}

    for row_number in range(len(table)):
        name = table[name_column].iloc[row_number]
        sequence = table[sequence_column].iloc[row_number]

        # Blank rows as NaN
        if not isinstance(name, str):
            continue
        if not isinstance(sequence, str):
            continue

        sequences[name] = sequence

    return sequences


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Embeds everything. Gives back names, lengths, residues, pooled.
# residues is (count, longest, width) float16. Row i is good up to
# lengths[i] and zero after that. pooled is the mean of the real residues.
def embed_sequences(sequences, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)
    model.eval()
    model.to(device)

    names = sorted(sequences.keys())

    # output array is sized by the longest sequence
    longest = 0
    for name in names:
        if len(sequences[name]) > longest:
            longest = len(sequences[name])

    width = model.config.hidden_size
    residues = np.zeros((len(names), longest, width), dtype=np.float16)
    pooled = np.zeros((len(names), width), dtype=np.float16)
    lengths = np.zeros(len(names), dtype=np.int32)

    for start in range(0, len(names), BATCH_SIZE):
        batch = names[start:start + BATCH_SIZE]

        batch_sequences = []
        for name in batch:
            batch_sequences.append(sequences[name])

        encoded = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        encoded = encoded.to(device)

        with torch.no_grad():
            output = model(**encoded)

        hidden = output.last_hidden_state.to(torch.float32).cpu().numpy()

        for position in range(len(batch)):
            name = batch[position]
            length = len(sequences[name])
            row = start + position

            # ESM-2 sticks a start token on the front and an end token on
            # the back. Skip both so row j is residue j.
            real = hidden[position, 1:length + 1, :]

            residues[row, :length, :] = real.astype(np.float16)
            pooled[row, :] = real.mean(axis=0).astype(np.float16)
            lengths[row] = length

        print("    " + str(start + len(batch)) + " / " + str(len(names)))

    return names, lengths, residues, pooled


# Loads one cached npz back.
# lengths used to build the attention mask
def load_embeddings(npz_path):
    loaded = np.load(npz_path, allow_pickle=False)
    return {
        "names": loaded["names"].tolist(),
        "lengths": loaded["lengths"],
        "residues": loaded["residues"],
        "pooled": loaded["pooled"],
    }


# Runs every model over receptors and peptides.
def main():
    root = project_root()
    data_folder = os.path.join(root, "Data")
    out_folder = os.path.join(root, "embeddings")
    os.makedirs(out_folder, exist_ok=True)

    device = pick_device()
    print("device: " + device)

    for tag in MODELS:
        model_path = MODELS[tag]
        print("")
        print("model " + tag + " (" + model_path + ")")

        for kind in SOURCES:
            file_name = SOURCES[kind][0]
            name_column = SOURCES[kind][1]
            sequence_column = SOURCES[kind][2]

            # Already built it on an earlier run, so leave it be.
            out_path = os.path.join(out_folder, kind + "_" + tag + ".npz")
            if os.path.exists(out_path):
                print("  " + kind + " already cached, skipping")
                continue

            csv_path = os.path.join(data_folder, file_name)
            sequences = read_sequence_csv(csv_path, name_column, sequence_column)
            print("  " + kind + ", " + str(len(sequences)) + " sequences")

            names, lengths, residues, pooled = embed_sequences(
                sequences, model_path, device
            )

            np.savez_compressed(
                out_path,
                names=np.array(names),
                lengths=lengths,
                residues=residues,
                pooled=pooled,
            )
            print("  wrote " + out_path + "  " + str(residues.shape))


if __name__ == "__main__":
    main()
