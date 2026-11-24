# F20DL 2026/27 - Group 17

> **Comparing Machine Learning Approaches Across Tabular and Imaging Biomedical Data**

A modality-aware comparative study using protein-expression data (Mice Protein) and high-resolution blood-cell images (BloodMNIST+ 128×128).

---

## 👥 Team Members

| Name | Role |
|------|------|
| Nirup | Team Lead / Configuration Lead  |
| Nishan | Final Report Lead |
| Rayyan | Data Management Lead |
| Aashin | Data Management Lead |
| Aadhi | Presentation & Lead / Report |


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

DYRK1A_N | ITSN1_N | BDNF_N | NR1_N | pAKT_N | S6_N | Genotype |
----------|---------|--------|-------|--------|------|----------|
|0.503644 | 0.747789 | 0.390665 | 0.658293 | 0.349068 | 0.518536 | c-CS-s |
| 0.514617 | 0.695239 | 0.430126 | 0.669340 | 0.354986 | 0.504261 | c-CS-s |

### BloodMNIST+

BloodMNIST+ image examples and class distributions are explored in [`notebooks/02_bloodmnist_eda.ipynb`](notebooks/02_bloodmnist_eda.ipynb).

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
joblib==1.4.2
matplotlib==3.11.2
medmnist==3.0.2
numpy==2.5.5
pandas==3.1.1
scikit_learn==1.9.1
seaborn==0.13.2
tensorflow==2.21.0
ucimlrepo==0.0.7
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
| 1 | Creates required local data directories |
| 2 | Downloads Mice Protein CSV from UCI archive |
| 3 | Downloads BloodMNIST+ (128×128) `.npz` files from MedMNIST servers |
| 4 | Saves downloaded datasets under `data/raw/` |

## Course Requirements & Locations

### R2 — Data Analysis and Exploration 
> *Comprehensive preprocessing, cleaning, EDA with visualisation, feature selection, and clustering.*

**Inputs:** Raw 77 protein features (tabular) / Raw 128×128 pixel images  
**Outputs:** Cleaned datasets, PCA visualisations, K-Means cluster assignments, silhouette scores

| Task | Notebook Location |
|------|-------------------|
| Tabular EDA | [`notebooks/01_mice_protien_eda.ipynb`](notebooks/01_mice_protien_eda.ipynb) |
| Seperate PCA & Clustering | [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) |
| Image Preprocessing & EDA | [`notebooks/02_bloodmnist_eda.ipynb`](notebooks/02_bloodmnist_eda.ipynb) |

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


**Decision Tree Practical Application:**
> For the Mice Protein dataset, Decision Trees extract interpretable biological rules (e.g., "IF DYRK1A_N > 0.52 AND S6_N < 0.48 THEN class = c-CS-s"). This is directly valuable for biologists identifying protein-expression thresholds linked to specific genetic conditions, unlike the 'black-box' nature of CNNs on images.

Location: [`notebooks/03_traditional_ml_baselines.ipynb`](notebooks/03_traditional_ml_baselines.ipynb)

---

### R4: Neural Networks 
> *Implement custom MLP and CNN models and compare their performance on biomedical images.*

**Inputs (MLP):** BloodMNIST+ RGB images (resized in the MLP script)  
**Inputs (CNN):** 128×128×3 RGB images  
**Outputs:** Predicted class (1 of 8)

| Model | Input | Accuracy | Macro-F1 | Notes |
|-------|---------|----------|----------|-------|
| MLP | BloodMNIST+ (32x32 resized) | 0.852 | 0.836 | Fully connected custom neural network |
| ResNet-50 | BloodMNIST+ (128x128) | 0.956 | 0.955 | Pretrained CNN feature extractor |
| EfficientNetB5 | BloodMNIST+ (128x128) | 0.893 | 0.886 | Pretrained CNN feature extractor |

Location: [`notebooks/compare_models_imaging.ipynb`](notebooks/compare_models_imaging.ipynb)

Training scripts: [`MLP`](scripts/MLP.py), [`ResNet50`](scripts/ResNet50-tl.py), [`EfficientNetB5`](scripts/EfficientNetB5-tl.py).

Training histories are stored in [`results folder`](results/).

## Project Milestones

| Milestone | Target Week | Requirement | Status |
|-----------|-------------|-------------|--------|
| Dataset selection & D1 Pitch | Week 4 | R1 | ✅ Complete |
| Automated data pipeline built | Week 5-6 | R2 | ✅ Complete |
| EDA, Preprocessing & Clustering complete | Week 6-7 | R2 | ✅ Complete |
| Traditional ML baselines trained | Week 8-10 | R3 | ✅ Complete |
| MLP + CNN implemented | Week 8-10 | R4 | ✅ Complete |
| Evaluation & error analysis | Week 10-11 | R3/R4 | ✅ Complete |
| Final Report & Code frozen | Week 10-11 | All | ✅ Complete |
| D3 Presentation prep | Week 12 | All | ⬜ Pending |

---

## 📁 Repository Structure

```
├── data/ (after running download_data.py)
│   └── raw/
│       ├── mice_protein_expression.csv
│       └── bloodmnist_128.npz
├── figures/ (stores mice protein training and evaluation analysis images)
├── models/ (created after running any of the Ml/CNN scripts)
├── notebooks/
│   ├── 01_mice_protien_eda.ipynb
│   ├── 02_bloodmnist_eda.ipynb
│   ├── clustering.ipynb
│   ├── decision_trees.ipynb
│   ├── knn.ipynb
│   ├── compare_models.ipynb
│   └── compare_models_imaging.ipynb
├── scripts/
│   ├── download_data.py
│   ├── data_pipeline.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── MLP.py
│   ├── ResNet50-tl.py
│   └── EfficientNetB5-tl.py
├── results/
│   ├── figures/ (stores imaging models evaluation analysis images)
│   ├── MLP_history.csv
│   ├── ResNet50_history.csv
│   └── EfficientNetB5_history.csv
├── requirements.txt
├── .gitignore
└── README.md                   # ← You are here
```

## Development Process

- **Version Control:** GitHub, using commits and merges to manage team contributions.
- **Meetings:** Team coordination during project development.
- **Code Reviews:** Team members review and integrate contributions.
- **Reproducibility:** Fixed random seeds used in the model scripts where applicable. Dependencies are listed in `requirements.txt`.

---

## References

1. Higuera, A., Sharma, K., & Baker, S. (2015). Defining the transcriptional targets of Down syndrome reveals a role for NKX2-1. *PLOS ONE*, 10(7). [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0132185)
2. Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. *Scientific Data*, 10, 65. [Link](https://www.nature.com/articles/s41597-023-01974-x)
3. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Scientific Data*, 7, 97. [Link](https://www.nature.com/articles/s41597-020-0428-7)

---


