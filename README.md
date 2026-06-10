# Diffusion-Assisted Brain MRI Tumor Classification

## Overview

This project investigates the impact of diffusion-based MRI preprocessing on brain tumor classification. A diffusion model is trained on healthy brain MRI scans from the IXI dataset and then used to denoise tumor MRI images. The study compares three classification pipelines:

1. **Raw Images**
2. **Denoised Images**
3. **Mixed (Raw + Denoised) Images**

A DenseNet121 classifier is used to classify brain MRI images into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The project also uses Grad-CAM for model explainability and visualization.

---

# Objectives

* Train a diffusion model on healthy brain MRI scans.
* Apply diffusion-based denoising to tumor MRI images.
* Compare classification performance using:

  * Raw images
  * Denoised images
  * Mixed datasets
* Analyze whether diffusion preprocessing improves tumor classification.
* Visualize model attention using Grad-CAM.

---

# Datasets

## 1. IXI Dataset

Used for diffusion model training.

Dataset Format:

```text
.nii
.nii.gz
```

Purpose:

* Learn healthy brain anatomy
* Train diffusion denoising model

---

## 2. Brain Tumor Dataset

Classes:

```text
glioma_tumor
meningioma_tumor
pituitary_tumor
no_tumor
```

Purpose:

* Brain tumor classification

---

# Project Structure

```text
project/

├── data/
│   ├── raw/
│   │   ├── IXI/
│   │   └── tumor_images/
│   │       ├── glioma_tumor/
│   │       ├── meningioma_tumor/
│   │       ├── pituitary_tumor/
│   │       └── no_tumor/
│   │
│   ├── processed/
│   │   ├── ixi_slices/
│   │   ├── tumor_images_processed/
│   │   ├── denoised_images/
│   │   └── synthetic_ixi/
│   │
│   └── splits/
│       ├── tumor_classification.csv
│       ├── tumor_classification_denoised.csv
│       └── tumor_classification_mixed.csv
│
├── outputs/
│   ├── diffusion/
│   ├── classifier/
│   ├── figures/
│   └── reports/
│
├── src/
│   ├── preprocess_ixi.py
│   ├── preprocess_tumor.py
│   ├── train_diffusion_ixi.py
│   ├── diffusion_denoise.py
│   ├── create_classification_csv.py
│   ├── make_mixed_csv.py
│   ├── dataset.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── gradcam_vis.py
│   ├── config.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

# Models Used

## Diffusion Model

### DDPM (Denoising Diffusion Probabilistic Model)

Purpose:

* Learn healthy MRI distributions
* Denoise tumor MRI images

Backbone:

* UNet

---

## Classification Model

### DenseNet121

Purpose:

* Brain tumor classification

Advantages:

* Feature reuse
* Strong performance on medical images
* Efficient parameter utilization

---

## Explainability

### Grad-CAM

Purpose:

* Visualize model attention
* Highlight tumor regions influencing predictions

---

# Methodology

## Step 1: IXI MRI Preprocessing

Input:

```text
IXI Dataset (.nii / .nii.gz)
```

Processing:

* Load MRI volume using nibabel
* Extract middle slices
* Normalize intensity
* Save slices as PNG

Output:

```text
data/processed/ixi_slices/
```

Run:

```bash
py src/preprocess_ixi.py
```

---

## Step 2: Train Diffusion Model

Input:

```text
IXI MRI slices
```

Model:

```text
DDPM + UNet
```

Output:

```text
outputs/diffusion/
```

Run:

```bash
py src/train_diffusion_ixi.py
```

---

## Step 3: Tumor Dataset Preprocessing

Input:

```text
Tumor images
```

Processing:

* Resize
* Normalize
* Organize class folders

Output:

```text
data/processed/tumor_images_processed/
```

Run:

```bash
py src/preprocess_tumor.py
```

---

## Step 4: Denoise Tumor Images

Input:

```text
Tumor images
```

Output:

```text
data/processed/denoised_images/
```

Run:

```bash
py src/diffusion_denoise.py
```

---

## Step 5: Create Dataset Splits

Generate:

```text
tumor_classification.csv
tumor_classification_denoised.csv
```

Run:

```bash
py src/create_classification_csv.py
```

---

## Step 6: Create Mixed Dataset

Combine:

* Raw training images
* Denoised training images

Run:

```bash
py src/make_mixed_csv.py
```

Output:

```text
tumor_classification_mixed.csv
```

---

# Classification Experiments

## Experiment 1: Raw Images

Run:

```bash
py src/train_classifier.py --exp_name improved_raw
```

Evaluate:

```bash
py src/evaluate_classifier.py --checkpoint outputs/classifier/improved_raw/best.pt
```

---

## Experiment 2: Denoised Images

Run:

```bash
py src/train_classifier.py --exp_name improved_denoised --use_denoised
```

Evaluate:

```bash
py src/evaluate_classifier.py --checkpoint outputs/classifier/improved_denoised/best.pt --use_denoised
```

---

## Experiment 3: Mixed Dataset

Run:

```bash
py src/train_classifier.py --exp_name improved_mixed --csv_name tumor_classification_mixed.csv
```

Evaluate:

```bash
py src/evaluate_classifier.py --checkpoint outputs/classifier/improved_mixed/best.pt --csv_name tumor_classification_mixed.csv
```

---

# Grad-CAM Visualization

Generate attention maps:

```bash
py src/gradcam_vis.py --checkpoint outputs/classifier/improved_raw/best.pt
```

Output:

```text
outputs/figures/gradcam/
```

---

# Experimental Findings

### Raw Images

* Highest classification accuracy
* Highest macro-F1 score
* Preserved tumor characteristics

### Denoised Images

* Significant performance degradation
* Healthy-brain priors suppressed tumor features

### Mixed Dataset

* Better than denoised-only
* Useful as diffusion-assisted augmentation

---

# Key Scientific Finding

Diffusion models trained on healthy brain MRI scans may suppress tumor-discriminative information when used as full-image denoisers.

This explains why:

```text
Raw > Mixed > Denoised
```

in classification performance.

---

# Future Work

* Conditional diffusion models
* Latent diffusion models
* Tumor-aware diffusion training
* Synthetic MRI generation
* Multi-modal MRI classification
* Vision Transformers (ViT)
* Ensemble learning

---

# Authors

Project: Diffusion-Assisted Brain MRI Tumor Classification

Technologies:

* Python
* PyTorch
* Diffusers
* DenseNet121
* DDPM
* UNet
* Grad-CAM
* Nibabel
* OpenCV

---

