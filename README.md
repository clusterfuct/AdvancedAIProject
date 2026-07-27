# Speech Emotion Recognition on RAVDESS  
### MFCC + SVM • Badshah CNN • ResNet‑18 • CAM Interpretability

This repository contains our Advanced AI course project on **Speech Emotion Recognition (SER)** using the **RAVDESS** dataset. We implement and compare three modeling approaches:

1. **MFCC + SVM baseline**  
2. **Reproduction of the Badshah et al. (2017) CNN architecture**  
3. **Fine‑tuned ResNet‑18 on Mel‑spectrograms**

We also include **CAM‑based interpretability**, extensive **dataset exploration**, and a unified **results comparison framework**.

---

## Project Overview

Speech Emotion Recognition aims to classify human emotions from audio signals.  
Using the RAVDESS dataset (1,440 labeled speech clips across 8 emotions), we:

- Perform waveform, spectrogram, silence, and metadata analysis  
- Extract MFCC, delta, and delta‑delta features  
- Train classical and deep learning models  
- Evaluate using accuracy, macro‑precision, macro‑recall, macro‑F1  
- Visualize confusion matrices and per‑class performance  
- Apply **Class Activation Mapping (CAM)** for interpretability  
- Compare all models in a unified results table

### **Final Model Performance**

| Model | Accuracy |
|-------|----------|
| **ResNet‑18 (Fine‑Tuned)** | **68%** |
| **MFCC + SVM** | **61%** |
| **Badshah CNN** | **40%** |

---

## Repository Structure
```
Repository Structure
├── notebooks/
│   └── Advanced_AI_Project.ipynb        # Full project notebook
├── data/
│   └── RAVDESS/                         # Place dataset here (not included)
├── models/
│   ├── best_badshah_cnn.pt              # Saved Badshah CNN weights
│   └── best_resnet18.pth                # Saved ResNet‑18 weights
├── README.md                            # Project documentation
└── requirements.txt                     # Python dependencies
```

---

## Dependencies

Create a `requirements.txt` with:

numpy
pandas
matplotlib
seaborn
librosa
soundfile
scikit-learn
torch
torchvision
torchaudio


Install manually:

```
# bash
pip install numpy pandas matplotlib seaborn librosa soundfile scikit-learn torch torchvision torchaudio
```

## Reproducibility Instructions
To reproduce our results exactly, set the random seed at the top of the notebook:

```
import random
import numpy as np
import torch

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
## Additional reproducibility steps
### 1. Deterministic MFCC + SVM
```
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
```
### 2. Deterministic ResNet‑18 DataLoader
```
g = torch.Generator()
g.manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    generator=g
)
```
## How to Run the Project
### 1. Download RAVDESS
Place the dataset under:
```
data/RAVDESS/

```
Or update the path in the notebook:
```
DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/DataSet")

```
### 2. Open the Notebook
Run:
```
notebooks/Advanced_AI_Project.ipynb

```
### 3. Execute Sections in Order

The notebook is organized as:

- **Dataset loading & exploration**
- **Waveform & spectrogram visualization**
- **Silence analysis & metadata checks**
- **MFCC extraction + SVM training**
- **Badshah CNN training & evaluation**
- **ResNet‑18 fine‑tuning**
- **CAM interpretability**
- **Unified results comparison**

---

### 4. View Outputs

You will see:

- **Classification reports**
- **Confusion matrices**
- **Per‑class F1 heatmaps**
- **CAM overlays**
- **Comparison tables**

---

## Key Results

- **MFCC + SVM outperformed the original Badshah CNN baseline**
- **ResNet‑18 achieved the best overall performance**
- **CAM visualizations showed deeper models attend to more discriminative spectral regions**
- **MFCC deltas and PCA improved classical model performance**
- **ResNet‑18 generalized best despite small dataset size**

---

## Interpretability

We applied **Class Activation Mapping (CAM)** to:

- **Badshah CNN** (conv3 layer)
- **ResNet‑18** (layer4 block)

This highlights which Mel‑spectrogram regions influenced predictions.

For **MFCC‑SVM**, we used:

- **Permutation importance**
- **Per‑class MFCC profile comparison**

---

## Citation

If you use this code, please cite:

- **RAVDESS dataset**
- **Badshah et al. (2017) CNN architecture**
- **ResNet‑18 (He et al., 2016)**

---

## Authors

- **Elizabeth Du Peza**  
- **Makayla McKenzie**  
- **Theresa Davis**  
- *Advanced AI — Spring 2026*
