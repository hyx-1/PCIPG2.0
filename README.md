Code for our multi-omics protein complex identification method PCIPG2.0. Datasets for PCIPG are in https://zenodo.org/records/18254107.
# PCIPG 2.0 — Structure-aware Multi-omics Unsupervised Protein Complex Discovery

PCIPG 2.0 is an end-to-end pipeline for **multi-omics unsupervised protein complex discovery** from **enhanced protein–protein interaction (PPI) networks** enhanced with **structure-derived residue embeddings**. The workflow (i) builds a multi-omics enhanced PPI subgraph, (ii) extracts per-residue structural embeddings using **ProteinMPNN**, (iii) trains an unsupervised GNN on the PPI graph, (iv) generates candidate complexes (top-*k* sets + clique seeds), and (v) evaluates predictions against gold-standard complexes.

## Quick Start
```bash
bash main.sh
```
