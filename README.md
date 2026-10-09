# F20DL 2026/27 - Group 17

> **Comparing Machine Learning Approaches Across Tabular and Imaging Biomedical Data**

A modality-aware comparative study using protein-expression data (Mice Protein) and high-resolution blood-cell images (BloodMNIST+ 128×128).

---

## 👥 Team Members

| Name | Role |
|------|------|
| Nirup | Team Lead / Configuration Lead  |
| Nishan | Final Report Lead |
| Rayyan | Configuration Management Lead |
| Aashin | Data Management Lead |
| Aadhi | Presentation & Report Lead |

---

## 🎯 Research Objectives

- Investigate how **model choice** affects multi-class classification across fundamentally different biomedical data modalities.
- Determine whether interpretable traditional ML (Decision Trees, k-NN, Naïve Bayes) is sufficient for high-dimensional tabular biological data.
- Evaluate whether Convolutional Neural Networks (CNNs) are necessary to capture spatial hierarchies in 128×128 cellular morphology images.
- Provide a rigorous, fair comparison using consistent evaluation metrics (Macro-F1, confusion matrices) rather than simply maximising accuracy.

**Research Question:**
> *How does model choice affect multi-class classification across different biomedical data types?*

---

## 📦 Dataset Sources & Licences

### Dataset 1 — Mice Protein Expression (Tabular)

| Property | Details |
|----------|---------|
| **True Source** | UCI Machine Learning Repository |
| **Original Paper** | Higuera, A., Sharma, K., & Baker, S. (2015). *Defining the transcriptional targets of Down syndrome reveals a role for NKX2-1.* PLOS ONE |
| **Licence** | CC BY 4.0 |
| **Link** | [https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) |
| **Samples** | 1,080 |
| **Features** | 77 protein-expression features |
| **Classes** | 8 (genotype/behaviour combinations) |
| **Task** | Multi-class classification |
| **ML Community Use** | Classic benchmark for testing clustering, feature selection, and classification on high-dimensional, correlated biological data. Used in studies on Down syndrome mouse models. |

### Dataset 2 — BloodMNIST+ (Imaging)

| Property | Details |
|----------|---------|
| **True Source** | MedMNIST+ (derived from original CBC dataset by Acevedo et al., 2020) |
| **Original Paper** | Yang, J., et al. (2023). *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.* Scientific Data |
| **Licence** | CC BY-NC-SA 4.0 |
| **Link** | [https://medmnist.com/](https://medmnist.com/) |
| **Training Images** | ~11,959 |
| **Image Size** | 128 × 128 × 3 (RGB) |
| **Classes** | 8 (blood cell types) |
| **Task** | Multi-class image classification |
| **ML Community Use** | Standardised benchmark for evaluating lightweight 2D biomedical image classification models and CNN architectures. Widely used for transfer learning experiments. |

---

## 🖼️ Data Examples

### Mice Protein — 2 Sample Rows

| MouseID | DYRK1A_N | ITSN1_N | BDNF_N | NR1_N | pAKT_N | S6_N | Genotype |
|---------|----------|---------|--------|-------|--------|------|----------|
| 309_M | 0.503644 | 0.747789 | 0.390665 | 0.658293 | 0.349068 | 0.518536 | c-CS-s |
| 310_M | 0.514617 | 0.695239 | 0.430126 | 0.669340 | 0.354986 | 0.504261 | c-CS-s |

### BloodMNIST+ — 2 Sample Images

<!-- Replace these with actual image links from your repo -->
![Neutrophil Example](documentation/images/bloodmnist_neutrophil.png)
![Eosinophil Example](documentation/images/bloodmnist_eosinophil.png)

> *Left: Neutrophil (multi-lobed nucleus visible at 128×128). Right: Eosinophil (distinct granule texture). Both at 128×128 RGB.*

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/NPN26/F21DL-2026-27-UG-group-17
cd F21DL-2026-27-UG-group-17

# 2. Create and activate virtual environment
conda create -n f20dl python=3.11
conda activate f20dl

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` (versions pinned)

```text
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1
matplotlib==3.8.3
seaborn==0.13.2
torch==2.2.1
torchvision==0.17.1
Pillow==10.2.0
jupyter==1.0.0
medmnist==3.0.1
```

---

## Data Preparation Pipeline

Our datasets exceed the GitHub repository size limit, so raw data is **not** stored in the repo. Instead, we use an automated download script.

### How to run:

```bash
python scripts/download_data.py
```

### What the pipeline does:

| Step | Action |
|------|--------|
| 1 | Downloads BloodMNIST+ (128×128) `.npz` files from MedMNIST servers |
| 2 | Downloads Mice Protein CSV from UCI archive |
| 3 | Verifies checksums / file integrity |
| 4 | Extracts into `data/raw/` |
| 5 | Splits into train/val/test sets → `data/processed/` |

> **Important:** `data/raw/` and `data/processed/` are listed in `.gitignore`. Run the script before opening any notebooks.

---

## Course Requirements & Locations

### R2 — Data Analysis and Exploration 
> *Comprehensive preprocessing, cleaning, EDA with visualisation, feature selection, and clustering.*

**Inputs:** Raw 77 protein features (tabular) / Raw 128×128 pixel images  
**Outputs:** Cleaned datasets, PCA visualisations, K-Means cluster assignments, silhouette scores

| Task | Notebook Location |
|------|-------------------|
| Tabular EDA & Clustering | [`notebooks/01_tabular_eda_clustering.ipynb`](notebooks/01_tabular_eda_clustering.ipynb) |
| Image Preprocessing & EDA | [`notebooks/02_image_preprocessing.ipynb`](notebooks/02_image_preprocessing.ipynb) |

<!-- Add a figure here once complete -->
<!-- ![PCA Scatter](documentation/images/pca_scatter.png) -->

---

### R3 — Baseline Training and Evaluation 
> *At least 3 traditional ML algorithms evaluated with appropriate metrics. Decision Trees discussed for practical interpretability.*

**Inputs:** Scaled 77 protein features (tabular)  
**Outputs:** Predicted class (1 of 8 genotypes)

| Model | Accuracy | Macro-F1 | Notes |
|-------|----------|----------|-------|
| Decision Tree | TBD | TBD | Interpretable biological rules |
| k-NN | TBD | TBD | Distance-based, requires scaling |
| Naïve Bayes | TBD | TBD | Probabilistic baseline |
| Random Forest | TBD | TBD | Ensemble comparison |

**Decision Tree Practical Application:**
> For the Mice Protein dataset, Decision Trees extract interpretable biological rules (e.g., "IF DYRK1A_N > 0.52 AND S6_N < 0.48 THEN class = c-CS-s"). This is directly valuable for biologists identifying protein-expression thresholds linked to specific genetic conditions, unlike the 'black-box' nature of CNNs on images.

Location: [`notebooks/03_traditional_ml_baselines.ipynb`](notebooks/03_traditional_ml_baselines.ipynb)

---

### R4 — Neural Networks 
> *Implement MLP and CNN models. Train, fine-tune, and compare against traditional ML.*

**Inputs (MLP):** Scaled 77 protein features  
**Inputs (CNN):** 128×128×3 RGB images  
**Outputs:** Predicted class (1 of 8)

| Model | Dataset | Accuracy | Macro-F1 | Notes |
|-------|---------|----------|----------|-------|
| MLP | Mice Protein | TBD | TBD | Tabular neural baseline |
| Custom CNN | BloodMNIST+ | TBD | TBD | Spatial feature extraction |
| ResNet-18 (fine-tuned) | BloodMNIST+ | TBD | TBD | Transfer learning |

Location: [`notebooks/04_neural_networks.ipynb`](notebooks/04_neural_networks.ipynb)

<!-- Add a figure here once complete -->
<!-- ![Training Curves](documentation/images/training_curves.png) -->

---

## Project Milestones

| Milestone | Target Week | Requirement | Status |
|-----------|-------------|-------------|--------|
| Dataset selection & D1 Pitch | Week 4 | R1 | ✅ Complete |
| Automated data pipeline built | Week 5 | R2 | 🔄 In Progress |
| EDA, Preprocessing & Clustering complete | Week 6 | R2 | ⬜ Pending |
| Traditional ML baselines trained | Week 7 | R3 | ⬜ Pending |
| MLP + CNN implemented | Week 9 | R4 | ⬜ Pending |
| Evaluation & error analysis | Week 10 | R3/R4 | ⬜ Pending |
| Final Report & Code frozen | Week 11 | All | ⬜ Pending |
| D3 Presentation prep | Week 12 | All | ⬜ Pending |

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                    # Auto-downloaded (in .gitignore)
│   └── processed/              # Train/val/test splits (in .gitignore)
├── notebooks/
│   ├── 01_tabular_eda_clustering.ipynb
│   ├── 02_image_preprocessing.ipynb
│   ├── 03_traditional_ml_baselines.ipynb
│   └── 04_neural_networks.ipynb
├── scripts/
│   ├── download_data.py        # Automated data pipeline
│   └── train.py                # Model training script
├── documentation/
│   ├── images/                 # Saved figures for README/report
│   └── meetings/               # Weekly meeting notes
├── requirements.txt
├── .gitignore
└── README.md                   # ← You are here
```

---

## Development Process

- **Version Control:** GitHub. All work committed via feature branches → Pull Requests → Merge.
- **Meetings:** Weekly during lab sessions. Notes stored in `documentation/meetings/`.
- **Code Reviews:** All PRs require at least one review before merging.
- **Reproducibility:** Random seeds set (`torch.manual_seed(42)`, `np.random.seed(42)`). All dependencies pinned in `requirements.txt`.

---

## References

1. Higuera, A., Sharma, K., & Baker, S. (2015). Defining the transcriptional targets of Down syndrome reveals a role for NKX2-1. *PLOS ONE*, 10(7). [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0132185)
2. Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. *Scientific Data*, 10, 65. [Link](https://www.nature.com/articles/s41597-023-01974-x)
3. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Scientific Data*, 7, 97. [Link](https://www.nature.com/articles/s41597-020-0428-7)

---

## Generative AI Disclaimer

Where Generative AI has been used to assist with code generation (permitted under Framework Statement 6 of the course policy), this is explicitly noted in code comments with the full prompt used. All AI-generated code has been independently evaluated and verified by the contributing team member.

---

*Last updated: 9 October 2026 • Maintained by: Nirup Nandish*
