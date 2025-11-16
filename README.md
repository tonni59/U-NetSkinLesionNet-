
---

## 🎯 Project Overview

Skin cancer (especially melanoma) requires **early and accurate detection** to improve survival rates.  
This research proposes a **hybrid CNN + Transformer + GAN-based methodology**:

- **K-means Segmentation** → initial lesion isolation  
- **U-Net++ Segmentation** → medical-grade boundary refinement  
- **CycleGAN** → synthetic malignant image generation  
- **Multiple CNN & Transformer Models** → lesion classification  
- **GradCAM++** → explainability and heatmap visualization  

---

## 🧠 Methodology

### 1️⃣ Segmentation Pipeline  
- Resize → preprocess  
- Apply K-means clustering  
- Smooth + threshold using Otsu  
- Merge ROI  
- Apply U-Net++ for enhanced segmentation  

### 2️⃣ CycleGAN Augmentation  
The CycleGAN architecture consists of:

- Generator G(A→B)  
- Generator G(B→A)  
- Discriminator D(A)  
- Discriminator D(B)  

Loss functions used:
- Adversarial Loss  
- Cycle-Consistency Loss  
- Identity Loss  

### 3️⃣ Classification Models Used  
- ResNet50  
- InceptionV3  
- Xception (via TIMM)  
- EfficientNetV2-L / V2-M  
- ConvNeXt Base  
- Swin Transformer-B  
- Vision Transformer (ViT-B/16)  
- **Custom Hybrid CNN-V model**  

---

## 🗂 Dataset

We used the **ISIC Skin Cancer Dataset (Malignant vs Benign)**.

**Dataset folder structure:**

