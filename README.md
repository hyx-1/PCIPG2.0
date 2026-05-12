# PCIPG2.0

Code for **PCIPG2.0**, a structure-aware and multi-omics-guided framework for unsupervised protein complex identification. PCIPG2.0 is a protein complex identification pipeline that supports both the original workflow and an enhanced workflow with multi-omics fusion. The enhanced workflow first refines the PPI network through multi-omics integration, and then runs the main prediction pipeline on the updated network.

Processed datasets used in this work are available at:  
**Zenodo:** https://sandbox.zenodo.org/records/468457

---

## Overview

**PCIPG2.0** is an end-to-end pipeline for unsupervised protein complex discovery from protein–protein interaction (PPI) networks. It combines:
    
- **multi-omics-driven interactome enhancement**, used to prioritize high-confidence candidate missing edges in the observed PPI network;
- **structure-aware protein representation learning**, based on residue-level embeddings extracted from protein structures;
- **PPI-level graph autoencoding**, used to infer latent complex memberships without complex labels.

Unlike methods that rely only on network topology, PCIPG2.0 explicitly addresses two common limitations of PPI-based complex discovery:

1. **missing interactions** in experimentally measured interactomes;
2. **limited mechanistic specificity** when protein nodes are modeled without residue- or structure-level information.

The method is fully unsupervised and is designed for robust protein complex identification under incomplete and noisy interactome measurements.

This repository provides two ways to run the project:

- **Original pipeline**: run the main model directly on the input PPI dataset
- **Enhanced pipeline with multi-omics fusion**: first perform multi-omics fusion to enhance the PPI network, then run the main pipeline on the enhanced PPI network

For the `collins` dataset, the enhanced workflow can be launched with a single command:

```bash
bash run_all.sh collins
```

---

## Method summary

The PCIPG2.0 workflow contains two main stages.

### 1. Multi-omics-guided PPI network enhancement
Complementary omics views are fused into a shared latent space. Protein pairs that are highly similar in the fused embedding space are treated as high-confidence candidate associations and can be added to the observed PPI graph as candidate missing edges.

Importantly, the multi-omics module is used **only to augment edge information where supported by omics similarity**. Proteins lacking matched multi-omics features are **not removed** from the network; their original interactions are retained unchanged.

### 2. Structure-aware complex inference
For each protein, residue-level structural embeddings are extracted using the **ProteinMPNN encoder** and aggregated on residue contact graphs to produce protein-level node features. These structure-aware node representations are then integrated with the enhanced PPI graph in an unsupervised **graph autoencoder** (GIN encoder + inner-product decoder) to infer latent complex memberships and generate predicted protein complexes.

---

## Main components

The repository includes code for:

- preprocessing benchmark PPI datasets;
- multi-omics feature integration and interactome enhancement;
- residue-graph construction from protein structures;
- ProteinMPNN-based residue embedding extraction;
- structure-aware protein representation learning;
- unsupervised PPI-level graph autoencoder training;
- complex extraction and redundancy removal;
- evaluation against gold-standard complexes;
- sensitivity and ablation analysis.

---

## Data availability

All processed data used in PCIPG2.0 are available at:

- **Zenodo:** https://sandbox.zenodo.org/records/468457

These resources include benchmark-specific processed inputs used in our experiments.

### Multi-omics source
Multi-omics functional features were derived from the Gene Expression Omnibus:

- **GEO accession:** `GSE168699`

The omics data include transcriptomic and chromatin-related profiles used for candidate-edge prioritization during interactome enhancement.

### Benchmark PPI datasets
PCIPG2.0 was evaluated on five widely used yeast PPI benchmarks:

- Collins
- Krogan-core
- Krogan14k
- DIP
- BioGRID

### Gold-standard complexes
The reference complex set was assembled from curated yeast complex resources, including:

- MIPS
- CYC2008
- SGD
- Aloy
- TAP06

---

## Project Structure

A simplified structure of the project is shown below:

```bash
PCIPG2/
├── environment.yml
├── main.sh
├── run_all.sh
├── PPI_Dataset/
│   └── collins.txt
├── multi_omics/
│   ├── input/
│   ├── output/
│   └── run_pipeline.py
└── ...
```

### Key Files

- `main.sh`  
  Runs the original main pipeline.

- `run_all.sh`  
  Runs the full enhanced workflow:
  1. multi-omics fusion
  2. copy the enhanced PPI file into the main pipeline directory
  3. run the main pipeline

- `multi_omics/run_pipeline.py`  
  Performs multi-omics fusion and generates an enhanced PPI network.

- `multi_omics/input/`  
  Stores the input files required by the multi-omics fusion module.

- `multi_omics/output/`  
  Stores the enhanced PPI files generated by the multi-omics fusion module.

- `PPI_Dataset/`  
  Stores the PPI datasets used by the main pipeline.

---

## Requirements

- Linux
- Conda
- Python environment defined in `environment.yml`

---

## Installation

Create the conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate pytorch
```

---

## Quick Start

### Run the Enhanced Pipeline with Multi-Omics Fusion

Using the `collins` dataset as an example:

```bash
bash run_all.sh collins
```

This command will automatically complete the following steps:

1. Run the multi-omics fusion pipeline
2. Save the enhanced PPI file to `multi_omics/output/collins.txt`
3. Copy the enhanced PPI file to `PPI_Dataset/collins.txt`
4. Run the main pipeline through `main.sh`

---

## Run the Original Pipeline Only

If you do **not** want to use multi-omics fusion, run:

```bash
bash main.sh
```

This executes the original workflow directly.

---

## How `run_all.sh` Works

The `run_all.sh` script is organized as follows:

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

DATASET="${1:-collins}"

echo "=== Step 1: Run multi-omics fusion ==="
python multi_omics/run_pipeline.py "$DATASET"

echo "=== Step 2: Copy enhanced PPI to main pipeline directory ==="
cp "multi_omics/output/${DATASET}.txt" "PPI_Dataset/${DATASET}.txt"

echo "=== Step 3: Run main pipeline ==="
bash main.sh

echo "=== All finished for dataset: ${DATASET} ==="
```

By default, if no dataset name is provided, the script uses `collins`.

For example, the following two commands are equivalent:

```bash
bash run_all.sh
```

```bash
bash run_all.sh collins
```

---

## Change to Another Dataset

If you want to use another dataset instead of `collins`, you need to update both the multi-omics module and the main pipeline.

### Step 1. Replace the Input Files in `multi_omics/input`

Put the corresponding files for your new dataset under:

```bash
PCIPG2/multi_omics/input/
```

Make sure the file names, formats, and organization are consistent with what the multi-omics pipeline expects.

### Step 2. Update Dataset Paths in `main.sh`

You also need to manually modify all dataset-related paths in:

```bash
PCIPG2/main.sh
```

Please check every Python command in `main.sh` and replace the paths that still point to the old dataset.

In other words, changing the dataset is **not only** passing a new argument to `run_all.sh`. You must also make sure that:

- the required input files in `multi_omics/input/` have been replaced
- all dataset paths referenced in `main.sh` have been updated accordingly

### Example

If you want to switch from `collins` to another dataset, the general workflow is:

1. prepare the new dataset files in `multi_omics/input/`
2. update all dataset-related paths in `main.sh`
3. run:

```bash
bash run_all.sh <your_dataset_name>
```

---

## Input and Output

### Multi-Omics Fusion Output

After running the multi-omics fusion module, the enhanced PPI file will be written to:

```bash
multi_omics/output/<dataset>.txt
```

### Main Pipeline Input

The enhanced PPI file is then copied to:

```bash
PPI_Dataset/<dataset>.txt
```

This copied file is used by the main pipeline.

---

## Recommended Usage

### Case 1: Run the Original Baseline

Use this if you want to evaluate the model without multi-omics enhancement:

```bash
bash main.sh
```

### Case 2: Run the Enhanced Version

Use this if you want to enhance the PPI network with multi-omics fusion before prediction:

```bash
bash run_all.sh collins
```

---

## Notes

- The default dataset in `run_all.sh` is `collins`
- Running `bash run_all.sh collins` is the recommended one-command entry point for the enhanced workflow
- When switching to a new dataset, editing `main.sh` is required
- The multi-omics input files must also be replaced before running a new dataset

---

## Troubleshooting

### 1. The Script Runs but Still Uses the Old Dataset

This usually means that some dataset paths in `main.sh` were not updated.  
Please check all Python commands inside `main.sh` and make sure every dataset-related path points to the correct files.

### 2. Multi-Omics Fusion Fails on a New Dataset

Please verify that:

- the required input files have been placed under `multi_omics/input/`
- the file names and formats match what `multi_omics/run_pipeline.py` expects

### 3. The Enhanced PPI File Is Not Used by the Main Pipeline

Please confirm that the copy step completed successfully and that the following file exists:

```bash
PPI_Dataset/<dataset>.txt
```
