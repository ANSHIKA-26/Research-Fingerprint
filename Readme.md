# Robust Latent Fingerprint Enhancement using Conditional GAN

This repository contains the implementation and experimental analysis of a **Conditional GAN (cGAN)–based latent fingerprint enhancement pipeline**, developed as part of a B.Tech research project.  
The work focuses on improving ridge clarity, orientation consistency, and structural quality of **real-world latent fingerprints**, which are typically noisy, partial, and degraded.

---

## Problem Statement

Latent fingerprints recovered from crime scenes often exhibit:
- Low ridge–valley contrast
- Broken or incomplete ridge structures
- Background noise, smudging, and distortion
- Partial impressions

Conventional fingerprint enhancement techniques (Gabor filters, FFT-based methods, KNN interpolation) and systems like AFIS are optimized for **rolled or plain fingerprints** and do not generalize well to latent inputs.

Even recent deep learning–based enhancement methods face limitations due to:
- Dependence on synthetic datasets
- Inconsistent ridge reconstruction
- Poor robustness across real-world latent variations

This project explores a **conditional generative approach** to learn a direct mapping from latent fingerprints to enhanced representations while preserving structural validity.

---

## Dataset

### Input Dataset
- **NIST SD302e**
  - ~9,990 real latent fingerprint images
  - Noisy, partial, crime-scene–like impressions
  - Used as generator input

### Reference Dataset
- **NIST SD302g**
  - Rolled and plain fingerprints
  - Used as conditional reference during training and evaluation

Ground-truth association is performed using **ID-based mapping**, followed by systematic pairing where strict one-to-one correspondence is unavailable.

---

## Implementation Details

- **Language:** Python  
- **Deep Learning Framework:** PyTorch / TensorFlow  
- **Model Architecture:** Conditional GAN (U-Net Generator + PatchGAN Discriminator)  
- **Image Processing:** OpenCV (grayscale conversion, resizing, CLAHE)  
- **Numerical & Visualization:** NumPy, Matplotlib  
- **Development Environment:** Jupyter Notebook  

---

## Preprocessing Pipeline

All fingerprint images pass through a standardized preprocessing pipeline:

1. **Grayscale Conversion**
   - Removes RGB channel noise
   - Forces the model to focus on ridge patterns

2. **Resolution Normalization**
   - Standardized to 1200 ppi
   - Ensures consistent ridge spacing

3. **Resizing**
   - Fixed resolution: `512 × 512`
   - Improves training stability and memory handling

4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Enhances local ridge–valley contrast
   - Suppresses background noise
   - Prevents over-enhancement via contrast limiting

---

## Model Architecture

### Conditional GAN (cGAN)

#### Generator
- U-Net–based encoder–decoder architecture
- Encoder captures ridge flow, orientation fields, and structural patterns
- Decoder reconstructs enhanced fingerprints
- Skip connections preserve high-frequency ridge and minutiae details

#### Discriminator
- PatchGAN discriminator
- Operates on local image patches
- Enforces texture realism and local ridge consistency

The generator is conditioned directly on the latent fingerprint input to reduce structural hallucination.

---

## Training Setup

- Dataset split:
  - 70% training
  - 15% validation
  - 15% testing
- Subject-wise separation enforced to prevent identity leakage
- Optimizer: Adam
- Loss components:
  - Adversarial loss for realism
  - Reconstruction loss for structural consistency

Qualitative outputs are saved across epochs to monitor noise suppression, ridge continuity, and orientation stabilization.

---

## Evaluation Strategy

Performance comparison is carried out between:
- **KNN-based enhancement**
- **GAN-based enhancement (proposed)**

Evaluation focuses on **structural fingerprint quality**, using metrics derived from:
- Ridge–valley contrast
- Orientation consistency
- Ridge strength

This prioritizes ridge alignment and continuity over raw pixel similarity.

---

## Observations

- GAN-based enhancement produces clearer ridge patterns and more consistent orientation flow
- Improved robustness to background noise and partial prints
- KNN-based enhancement suffers from over-smoothing and instability on degraded inputs

Overall, the conditional GAN approach demonstrates stronger structural enhancement for latent fingerprints compared to classical interpolation-based methods.
