# 🧬 Geneformer-CAB: Benchmarking Scale and Architecture in Foundation Models for Single-Cell Transcriptomics

## Model Overview

**Geneformer-CAB (Cumulative-Assignment-Blocking)** is a benchmarked variant of the Geneformer architecture for modeling single-cell transcriptomic data.  
Rather than introducing an entirely new model, Geneformer-CAB systematically evaluates how **data scale** and **architectural refinements** interact to influence model generalization, predictive diversity, and robustness to batch effects.

This model integrates two architectural enhancements:
- **Cumulative probability recalibration**, which adjusts token-level prediction dynamics to reduce overconfident, frequency-driven outputs.  
- **Similarity-based regularization**, which penalizes redundant token predictions to promote diversity and alignment with rank-ordered gene expression profiles.

Together, these mechanisms provide insight into the **limits of scale** in single-cell foundation models — revealing that scaling up pretraining data does not always yield superior downstream performance.

---

## Key Results

| Task Type | Comparison | Key Finding |
|------------|-------------|-------------|
| **Pretraining Objectives** | GF-CAB vs. Geneformer | Higher masked prediction accuracy and diversity across scales |
| **Classification Tasks** | GF-CAB-1M vs. Geneformer-1M | Comparable or improved accuracy, narrowing the scale gap |
| **Zero-shot Batch Mitigation** | GF-CAB vs. Geneformer | Stronger generalization across datasets, less scale-dependent |

> Scaling pretraining data from 1M to 30M profiles improved discriminative tasks but reduced cross-dataset robustness — while architectural calibration in GF-CAB balanced both.

---

## Model Architecture

- **Base architecture:** Transformer encoder (BERT-style masked modeling)  
- **Input representation:** Ranked gene expression profiles per cell  
- **Masking objective:** Predict masked gene ranks, excluding unmasked regions  
- **Innovations:**  
  - Cumulative probability recalibration (adjusted decoding dynamics)  
  - Similarity-based penalty loss (reduces redundancy in token predictions)

---

## Pretraining Data

| Dataset | Description | Size |
|----------|--------------|------|
| **Genecorpus-1M** | Random subset of ranked single-cell profiles from public scRNA-seq datasets | 1 million profiles |
| **Genecorpus-30M** | Large-scale extension incorporating additional datasets and donors | 30 million profiles |

---

## Downstream Evaluation

1. **Cell-type classification** (3 benchmark tasks)  
2. **Zero-shot batch-effect mitigation** (4 public datasets)

Evaluation followed standardized pipelines based on Theodoris et al. (for classification) and Kedzierska et al. (for zero-shot robustness).

---

## Intended Use

This model is designed for:
- Benchmarking **foundation models** on single-cell gene expression tasks  
- Studying **scaling effects** in biological pretraining  
- Investigating **rank-based profile modeling** and representation diversity  

---

## Model details can be found on Huggingface repositories

- Script & Trained models: [https://huggingface.co/JFLa/GF-CAB].

- Datasets: [https://huggingface.co/datasets/JFLa/GF-CAB_Datasets].
