"""
Fetch the parts of the DeorphaNN dataset we need.

The repo is 77.7 GB. Most of it is AlphaFold output we do not use. We take the
label CSV plus 403 PDB files, about 182 MB.

Sequences are not stored as FASTA. They only exist inside the PDB files. The
data is a full 65 x 339 grid, so one receptor folder gives every peptide, and
one file per other receptor gives the rest.
"""

import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO_ID = "lariferg/DeorphaNN"
API_URL = "https://huggingface.co/api/datasets/" + REPO_ID
FILE_URL = "https://huggingface.co/datasets/" + REPO_ID + "/resolve/main/"

LABELS_PATH = "DeorphaNN_training/Dataset_Labels_full - Sheet1.csv"

# Raise this on Turing. The network there is faster.
NUM_WORKERS = 16


def build_url(repo_path):
    """Make a download URL from a path inside the repo."""
    return FILE_URL + urllib.parse.quote(repo_path)


def download_one_file(repo_path, local_path):
    """Download one file. Skips files already on disk, so reruns are cheap."""
    if os.path.exists(local_path):
        return "skipped"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Write to a .part name first. A killed run then leaves no file that looks
    # finished.
    temp_path = local_path + ".part"
    urllib.request.urlretrieve(build_url(repo_path), temp_path)
    os.replace(temp_path, local_path)
    return "downloaded"


def list_repo_files():
    """Return every file path in the repo."""
    response = urllib.request.urlopen(urllib.request.Request(API_URL), timeout=60)
    info = json.load(response)
    response.close()

    paths = []
    for entry in info["siblings"]:
        paths.append(entry["rfilename"])
    return paths


def group_pdbs_by_receptor(all_paths):
    """
    Sort PDB paths into buckets keyed by receptor.

    Paths look like relaxed_PDBs/AEX-2-1/amber_r_AEX-2-1_FLP-1-1_...pdb
    so the folder after relaxed_PDBs is the receptor.
    """
    buckets = {}
    for path in all_paths:
        if not path.startswith("relaxed_PDBs/"):
            continue
        if not path.endswith(".pdb"):
            continue

        parts = path.split("/")
        if len(parts) < 3:
            continue

        receptor = parts[1]
        if receptor not in buckets:
            buckets[receptor] = []
        buckets[receptor].append(path)

    # Sort so the pick is the same on every run.
    for receptor in buckets:
        buckets[receptor].sort()
    return buckets


def choose_minimal_pdb_set(buckets):
    """
    Pick the fewest PDB files that still cover every sequence.

    Take all files from the biggest receptor folder for the peptides. Take one
    file from each other receptor.
    """
    peptide_source = None
    largest = -1
    for receptor in sorted(buckets.keys()):
        count = len(buckets[receptor])
        if count > largest:
            largest = count
            peptide_source = receptor

    chosen = []
    for path in buckets[peptide_source]:
        chosen.append(path)

    for receptor in sorted(buckets.keys()):
        if receptor == peptide_source:
            continue
        chosen.append(buckets[receptor][0])

    return chosen, peptide_source


def download_dataset(data_folder):
    """Download the label CSV and the PDB subset. Returns the PDB folder."""
    os.makedirs(data_folder, exist_ok=True)

    print("Downloading labels.")
    labels_local = os.path.join(data_folder, "labels.csv")
    print("  labels.csv " + download_one_file(LABELS_PATH, labels_local))

    print("Listing repo files.")
    buckets = group_pdbs_by_receptor(list_repo_files())
    chosen, peptide_source = choose_minimal_pdb_set(buckets)

    print("Receptors: " + str(len(buckets)))
    print("Peptide source: " + peptide_source)
    print("Need " + str(len(chosen)) + " PDB files.")

    pdb_folder = os.path.join(data_folder, "pdb")

    jobs = []
    for repo_path in chosen:
        parts = repo_path.split("/")
        local_path = os.path.join(pdb_folder, parts[1], parts[-1])
        jobs.append((repo_path, local_path))

    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    futures = []
    for job in jobs:
        futures.append(pool.submit(download_one_file, job[0], job[1]))

    done = 0
    for future in futures:
        future.result()
        done = done + 1
        if done % 100 == 0:
            print("  " + str(done) + " / " + str(len(jobs)))
    pool.shutdown()

    return pdb_folder
