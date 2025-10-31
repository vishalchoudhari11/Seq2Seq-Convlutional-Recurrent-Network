# Sequence-to-Sequence Modeling with Convolutional Recurrent Networks (CRN)

This repository demonstrates a **sequence-to-sequence modeling framework** based on **Convolutional Recurrent Networks (CRNs)** — a hybrid architecture combining convolutional feature extraction, recurrent temporal modeling, and frequency-domain normalization.

While originally developed for **speech enhancement**, this framework is general and can be applied to **any time–frequency sequence reconstruction** problem (e.g., biosignals, sensor denoising, or other temporal modalities).

---

## 🚀 Key Features

- 🧩 **Modular PyTorch implementation**
  - `TemporalCRN` — a compact yet powerful Conv+LSTM sequence model.
  - Streaming-friendly architecture with causal convolutions and per-frame normalization.
- 🔄 **Fully functional training & evaluation pipeline**
  - Dataset, DataLoader, and training utilities included.
  - Combined waveform + STFT magnitude loss.
- 🎛️ **Configurable parameters**
  - Adjustable hop/window lengths, overlap-add synthesis, and target sample rates.
- 📈 **Experiment tracking**
  - Automatic CSV logging, checkpoint saving, and loss curve visualization.
- 🎧 **Application-ready**
  - Converts complex-domain STFT predictions back to waveform using overlap-add synthesis.

---

## 🗂️ Repository Overview

| Folder / File | Description |
|----------------|-------------|
| `src/model.py` | CRN architecture (encoder–decoder with LSTM core). |
| `src/dataset.py` | Sequence dataset loader with SNR-controlled mixing and STFT computation. |
| `src/train_utils.py` | Core training & evaluation utilities. |
| `src/losses.py` | Waveform and STFT-based loss functions. |
| `src/utils/` | I/O, windowing, and inverse STFT utilities. |
| `src/main.py` | Entry point for model training and evaluation. |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/vishalchoudhari11/Seq2Seq-Convolutional-Recurrent-Network.git
cd seq2seq-crn

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧰 Dependencies

All dependencies are standard PyTorch + scientific Python libraries.

```text
torch
torchaudio
numpy
pandas
matplotlib
soundfile
scipy
```

---

## 🧑‍💻 Usage

### 1️⃣ Prepare your dataset
Place your dataset under:
```
data/
├── Source_16kHz/
└── Auxiliary_16kHz/
```

Each partition (`train`, `dev`, `test`) should follow:
```
data/Source_16kHz/train/
data/Auxiliary_16kHz/train/
```

Here, the `Auxiliary_16kHz` folder can hold a secondary source such as interfering noise or competing speech.


### 2️⃣ Train the model
```bash
python src/main.py
```

---

## 📚 References

This implementation is inspired by the foundational works introducing and extending **Convolutional Recurrent Networks (CRN)** for complex-domain sequence modeling:

1. **Tan, K. & Wang, D. (2018).**  
   *A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement.*  
   In *Proceedings of Interspeech 2018*, pp. 3229–3233. https://doi.org/10.21437/Interspeech.2018-1405  

2. **Tan, K. & Wang, D. (2019).**  
   “Complex Spectral Mapping with a Convolutional Recurrent Network for Monaural Speech Enhancement.”  
   In *ICASSP 2019 – IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 6865–6869. https://doi.org/10.1109/ICASSP.2019.8682834  
