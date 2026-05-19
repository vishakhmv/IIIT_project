# 🎭 Multimodal Emotion Recognition (Speech + Text Fusion)

## 📖 Description

This project implements a **Multimodal Emotion Recognition System** capable of predicting human emotions using **speech-only, text-only, and combined speech-text inputs**.  

The system classifies emotions into seven categories: **Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, and Sad**.

The repository contains three independent deep learning pipelines:

- 🎙️ **Speech-Only Model** — learns emotional patterns from acoustic speech features such as tone, pitch, and prosody.
- 📝 **Text-Only Model** — analyzes emotional meaning from the semantic and contextual understanding of transcribed text.
- 🔀 **Multimodal Fusion Model** — combines both speech and text representations to achieve stronger and more robust emotion recognition.

The project also includes **representation learning analysis, t-SNE visualization, confusion matrices, and error analysis** to study the effectiveness of multimodal learning.

## 📊 Dataset

The models are trained and evaluated on the **TESS (Toronto Emotional Speech Set)** dataset.

- **Dataset Link:** [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

- **Emotion Distribution:**

<p align="center">
  <img src="assets/dataset-distribution.png" width="70%">
</p>

---

### Dataset Overview

- Consists of **2,800 speech samples** recorded by two actresses aged **26** and **64**.
- Balanced across **7 emotional categories**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Pleasant Surprise
  - Sad

---

### Linguistic Structure

The dataset uses linguistically neutral carrier phrases such as:

> *“Say the word door”*  
> *“Say the word neat”*

All emotional categories contain the same spoken sentences, ensuring that emotional variation primarily comes from vocal expression rather than changes in textual content.

---

### Data Split

The dataset is divided using an **80% Training / 20% Testing** split while maintaining balanced emotion distributions during evaluation.


## 📦 Data Extraction & Splitting

The TESS dataset was extracted and processed using custom preprocessing scripts developed in Google Colab.

<p align="center">
  <img src="assets/tess-data-flow.png" width="85%">
</p>

---

### 🔓 Data Extraction

The compressed TESS dataset archive was extracted from Google Drive and verified before preprocessing.

### Extraction Pipeline

- Mounted Google Drive for dataset access
- Loaded the compressed dataset archive
- Extracted audio files using Python's `zipfile` utilities
- Verified extracted folder structure and audio contents

### Output Structure

```bash
/content/tess_dataset/
└── TESS Toronto emotional speech set data/


```
### ✂️ Data Splitting

After dataset extraction, the complete TESS dataset was recursively processed to generate structured metadata for multimodal training and testing.

The preprocessing pipeline created paired speech-text samples by extracting:

- Relative audio file paths
- Reconstructed text transcripts
- Encoded emotion labels

---

### 📝 Transcript Generation

The TESS dataset only contains audio files and does not provide separate text transcripts.  
Since the dataset follows fixed sentence patterns, text transcripts were generated directly from the audio filenames.

Example filename:

```text
YAF_back_angry.wav
```

This filename contains:

- `YAF` → Speaker ID  
- `back` → Spoken word  
- `angry` → Emotion label  

The preprocessing pipeline extracts the spoken word from the filename and reconstructs the transcript:

```python
word = name_parts[1].lower()
text_phrase = f"say the word {word}"
```

Generated transcript:

```text
YAF_back_angry.wav → "say the word back"
```

This generated text was used as input for:

- Text-Only Model
- Multimodal Fusion Model

### Example Speech-Text Pairs

| Audio File | Generated Text |
|---|---|
| YAF_back_angry.wav | say the word back |
| YAF_door_happy.wav | say the word door |

---

### 🏷️ Emotion Label Encoding

Each emotion category was mapped into a numerical label for model training.

| Emotion | Label |
|---|---|
| Angry | 0 |
| Disgust | 1 |
| Fear | 2 |
| Happy | 3 |
| Neutral | 4 |
| Pleasant Surprise | 5 |
| Sad | 6 |

---

### 📊 Train-Test Split

The complete dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

using a fixed random seed for reproducibility.

The split preserved balanced emotion distributions across both datasets to ensure fair evaluation for all emotion classes.

---

### 💾 Generated CSV Files

The splitting pipeline generated two structured CSV files:

```bash
Data_split/
├── train_split.csv
└── test_split.csv
```

### train_split.csv
Contains all training samples used for:

- Model learning
- Feature extraction
- Parameter optimization

### test_split.csv
Contains unseen testing samples used for:

- Final evaluation
- Accuracy measurement
- Generalization analysis

---

### 📁 CSV Structure

Each CSV file contains the following columns:

| Column | Description |
|---|---|
| path | Relative path to audio sample |
| text | Generated text transcript |
| label_encoded | Numerical emotion label |

---

### 📌 Example CSV Entry

| path | text | label_encoded |
|---|---|---|
| YAF_angry/YAF_back_angry.wav | say the word back | 0 |

---

### ✅ Outcome

The final preprocessing pipeline produced a clean and reproducible multimodal dataset structure suitable for:

- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion Training


## 🧠 Models

This project consists of three independent deep learning pipelines:

### 🎙️ 1. Speech-Only Model

The Speech-Only pipeline predicts emotions directly from raw audio signals by learning acoustic speech features such as tone, pitch, energy, etc.

### 🏋️ a. Training Pipeline

The Speech-Only training pipeline predicts emotions directly from raw audio signals using a hybrid deep learning architecture that combines pretrained speech representations, handcrafted acoustic features, and temporal sequence modeling.

---

### 🧠 Model Architecture

The training architecture integrates:

- **HuBERT** for contextual speech representation learning
- **MFCC Features** for acoustic feature extraction
- **BiLSTM Layers** for temporal emotional pattern modeling
- **Fully Connected Classification Head** for final emotion prediction

---

### 🔊 Audio Preprocessing

Each audio sample undergoes multiple preprocessing stages before training:

- Audio loading using `librosa`
- Silence trimming
- Fixed-length padding/truncation
- MFCC feature extraction
- Raw waveform preparation for HuBERT

All audio samples are standardized to a fixed duration for consistent batch training.

---

### 🎼 Feature Extraction

The model extracts two complementary speech representations from every audio sample.

### 1. HuBERT Embeddings

A pretrained **HuBERT (`facebook/hubert-base-ls960`)** model generates contextual speech embeddings directly from raw audio waveforms.

These embeddings capture:

- Speech context
- Prosody
- Temporal speech structure
- High-level acoustic patterns

The pretrained HuBERT weights are frozen during training to preserve learned speech representations.

---

### 2. MFCC Acoustic Features

MFCC features are extracted using `librosa` to capture low-level acoustic characteristics such as:

- Pitch
- Tone
- Frequency distribution
- Spectral variations

The MFCC features are then combined with HuBERT embeddings to create a richer emotional speech representation.

---

### 🔗 Feature Fusion

The extracted HuBERT embeddings and MFCC features are concatenated along the feature dimension to form a unified acoustic representation.

This fusion allows the model to learn from both:

- Deep contextual speech embeddings
- Traditional handcrafted acoustic features

simultaneously.

---

### ⏳ Temporal Sequence Modeling

The fused speech representation is passed through a **Bidirectional LSTM (BiLSTM)** network.

The BiLSTM learns temporal emotional dependencies across speech frames by modeling:

- Emotional transitions
- Speaking rhythm
- Sequential acoustic dynamics
- Prosodic variations over time

Global temporal pooling is applied after the BiLSTM to generate a compact emotional embedding.

---

### 🎯 Emotion Classification

The pooled emotional embedding is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier predicts one of the seven emotional categories.

---

### 📊 Dataset Loading

Training samples are loaded directly from:

```bash
Data_split/train_split.csv
```

The CSV file contains:

| Column | Description |
|---|---|
| path | Relative audio file path |
| text | Generated transcript |
| label_encoded | Numerical emotion label |

The training dataset is further divided into:

- Training Set
- Validation Set

to monitor generalization performance during learning.

---

### ⚙️ Optimization Strategy

The training pipeline uses:

- **AdamW Optimizer**
- **CrossEntropy Loss**
- **Dropout Regularization**
- **Validation-based Checkpoint Saving**

The best-performing model weights are automatically saved during training.

---

### 📈 Training Monitoring

During training, the pipeline tracks:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Learning curves are generated for performance visualization and convergence analysis.

### 💾 Generated Outputs

```bash
Results/
└── plots/
    └── Speech_model/
        └── learning_curves.png
```

The best-performing model weights obtained during validation are automatically saved as:

```bash
best_speech_model.pth
```

This file stores the learned parameters of the Speech-Only deep learning model and is later loaded during the testing and evaluation phase for inference on unseen samples.

### 💾 Storage Location

```bash
IIIT_project/
└── best_speech_model.pth
```
