# PCIPG2.0

Code for **PCIPG2.0**, a structure-aware and multi-omics-guided framework for unsupervised protein complex identification.

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

## Repository structure

A typical repository organization is as follows:

```text
PCIPG2.0/
├── data/                  # processed benchmark inputs and auxiliary files
├── omics/                 # multi-omics preprocessing and fusion
├── structure/             # residue-graph construction and structural embeddings
├── model/                 # GNN / autoencoder models
├── evaluation/            # evaluation scripts and metrics
├── scripts/               # training / inference / utility scripts
├── main.sh                # quick-start entry point
└── README.md

## Quick Start
```bash
bash main.sh
```
