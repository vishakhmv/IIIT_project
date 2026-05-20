# 🎭 Multimodal Emotion Recognition (Speech and Text)

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


## 📦 Data Extraction & Splitting

The TESS dataset was extracted and processed using custom preprocessing scripts developed in Google Colab.

<p align="center">
  <img src="assets/tess-data-flow.png" width="85%">
</p>

---

### 🔓 Data Extraction

The compressed TESS dataset archive was extracted from Google Drive and verified before preprocessing.

#### Extraction Pipeline

- Mounted Google Drive for dataset access
- Loaded the compressed dataset archive
- Extracted audio files using Python's `zipfile` utilities
- Verified extracted folder structure and audio contents

#### Output Structure

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

#### 📝 Transcript Generation

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

#### Example Speech-Text Pairs

| Audio File | Generated Text |
|---|---|
| YAF_back_angry.wav | say the word back |
| YAF_door_happy.wav | say the word door |

---

#### 🏷️ Emotion Label Encoding

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

#### 📊 Train-Test Split

The complete dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

using a fixed random seed for reproducibility.

The split preserved balanced emotion distributions across both datasets to ensure fair evaluation for all emotion classes.

---

#### 💾 Generated CSV Files

The splitting pipeline generated two structured CSV files:

```bash
Data_split/
├── train_split.csv
└── test_split.csv
```

#### train_split.csv
Contains all training samples used for:

- Model learning
- Feature extraction
- Parameter optimization

#### test_split.csv
Contains unseen testing samples used for:

- Final evaluation
- Accuracy measurement
- Generalization analysis

---

#### 📁 CSV Structure

Each CSV file contains the following columns:

| Column | Description |
|---|---|
| path | Relative path to audio sample |
| text | Generated text transcript |
| label_encoded | Numerical emotion label |

---

#### 📌 Example CSV Entry

| path | text | label_encoded |
|---|---|---|
| YAF_angry/YAF_back_angry.wav | say the word back | 0 |

---

#### ✅ Outcome

The final preprocessing pipeline produced a clean and reproducible multimodal dataset structure suitable for:

- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion Training


## 🧠 Models

This project consists of three independent deep learning pipelines:

### 🎙️ 1. Speech-Only Model

The Speech-Only pipeline predicts emotions directly from raw audio signals by learning acoustic speech features such as tone, pitch, energy, etc.

#### 🏗️ a. System Architecture

The Speech-Only architecture predicts emotions directly from raw speech audio by combining pretrained contextual speech embeddings, handcrafted acoustic features, and temporal sequence modeling.

<p align="center">
  <img src="assets/speech-model-architecture.png" width="90%">
</p>

---

##### 🔊 Input Audio

The model receives raw speech audio samples as input.

Each audio sample undergoes multiple preprocessing operations before feature extraction:

- Audio loading using `librosa`
- Silence trimming
- Fixed-length padding/truncation
- Sample rate normalization
- MFCC feature extraction preparation

To ensure consistent batch processing, all audio samples are standardized to a fixed duration and sampling rate.

---

##### 🎼 Parallel Feature Extraction

The architecture extracts two complementary speech representations simultaneously:

---

##### 1. HuBERT Contextual Speech Embeddings

The raw waveform is passed through a pretrained **HuBERT (`facebook/hubert-base-ls960`)** model.

HuBERT generates high-dimensional contextual speech embeddings that capture:

- Speech context
- Prosody
- Temporal speech structure
- High-level acoustic semantics
- Phonetic and emotional speech patterns

The pretrained HuBERT weights are frozen during training to preserve learned speech representations and reduce computational overhead.

---

##### 2. MFCC Acoustic Features

In parallel, MFCC (Mel-Frequency Cepstral Coefficient) features are extracted from the speech signal using `librosa`.

MFCC features capture low-level acoustic characteristics such as:

- Pitch
- Tone
- Frequency distribution
- Spectral variations
- Vocal tract characteristics

These handcrafted acoustic descriptors complement the contextual HuBERT representations.

---

##### 🔗 Feature Fusion

The HuBERT embeddings and MFCC acoustic features are concatenated together along the feature dimension to create a unified speech representation.

This fusion mechanism enables the architecture to simultaneously learn from:

- Deep contextual speech representations
- Traditional handcrafted acoustic features

The fused representation provides richer emotional information compared to using either representation independently.

---

##### ⏳ Temporal Sequence Modeling

The combined speech representation is passed through a **Bidirectional Long Short-Term Memory (BiLSTM)** network.

The BiLSTM models sequential emotional dependencies across speech frames and learns temporal speech dynamics such as:

- Emotional transitions
- Speaking rhythm
- Temporal prosodic variations
- Sequential acoustic patterns

Bidirectional processing allows the network to capture contextual dependencies from both forward and backward directions in the speech sequence.

---

##### 🌐 Global Temporal Pooling

After BiLSTM processing, global temporal pooling is applied across all time steps.

This operation compresses the sequential BiLSTM outputs into a compact fixed-dimensional emotional embedding representing the overall emotional characteristics of the speech sample.

---

##### 🎯 Emotion Classification Head

The pooled emotional embedding is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier predicts one of the following seven emotional categories:

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

##### 📤 Final Output

The final output of the architecture is a probability distribution across all emotion classes.

The emotion with the highest probability score is selected as the predicted emotional state of the input speech sample.

#### 🏋️ b. Training Pipeline

The Speech-Only training pipeline learns emotional representations directly from speech signals using supervised deep learning.

The complete training workflow includes:

- Dataset loading
- Audio preprocessing
- Acoustic feature extraction
- Temporal sequence learning
- Emotion classification
- Validation monitoring
- Checkpoint saving

---

##### 📂 Dataset Loading

Training samples are loaded from:

```bash
Data_split/train_split.csv
```

The CSV file contains:

| Column | Description |
|---|---|
| path | Relative audio file path |
| text | Generated transcript |
| label_encoded | Numerical emotion label |

The dataset is loaded using the custom `TESSSpeechDataset` class implemented in PyTorch.

---

##### ✂️ Training–Validation Split

The original training dataset is further divided into:

- Training Set
- Validation Set

using:

```python
train_test_split()
```

with stratified sampling to preserve balanced emotion distributions across both subsets.

The validation dataset is used to:

- Monitor generalization performance
- Track validation accuracy
- Prevent overfitting
- Save the best-performing model weights

---

##### 🔊 Audio Preprocessing

Before feature extraction, every audio sample undergoes multiple preprocessing stages.

The preprocessing pipeline performs:

- Audio loading using `librosa`
- Silence trimming
- Fixed-length padding/truncation
- Sample rate normalization
- MFCC feature extraction preparation

All speech samples are standardized to a fixed duration to ensure consistent batch processing during training.

---

##### 🎼 Feature Preparation

For each audio sample, the training pipeline generates two complementary speech representations.

##### 1. Raw Waveform Input

The raw speech waveform is directly passed into the pretrained HuBERT encoder for contextual speech representation learning.

##### 2. MFCC Acoustic Features

MFCC features are extracted using:

```python
librosa.feature.mfcc()
```

These features capture low-level acoustic properties such as:

- Pitch
- Tone
- Spectral characteristics
- Frequency distribution

The extracted MFCC features are later fused with contextual HuBERT embeddings inside the architecture.

---

##### 🔗 Batch Loading Pipeline

The processed samples are loaded using PyTorch `DataLoader` objects.

Two separate data loaders are created:

- Training Loader
- Validation Loader

```python
train_loader = DataLoader(...)
val_loader = DataLoader(...)
```

The training loader uses shuffled batches for randomized learning, while the validation loader preserves deterministic ordering during evaluation.

---

##### 🧠 Forward Propagation

During each training iteration:

1. Raw audio is passed through the pretrained HuBERT encoder
2. MFCC acoustic features are extracted
3. Both feature representations are fused together
4. The fused representation passes through BiLSTM layers
5. Temporal pooling generates compact emotional embeddings
6. The classifier predicts emotion probabilities

The final output is a probability distribution across all emotion classes.

---

##### 📉 Loss Function

The training pipeline uses:

```python
CrossEntropyLoss()
```

to measure prediction error between:

- Predicted emotion probabilities
- Ground-truth emotion labels

This loss function is suitable for multi-class emotion classification tasks.

---

##### ⚙️ Optimization Strategy

The model parameters are optimized using:

```python
torch.optim.AdamW()
```

The optimizer updates trainable parameters using gradient-based backpropagation.

Training configuration:

| Parameter | Value |
|---|---|
| Learning Rate | `1e-4` |
| Weight Decay | `1e-2` |
| Batch Size | `32` |
| Epochs | `10` |

---

##### 🔄 Backpropagation & Parameter Updates

For every training batch:

1. Forward propagation is performed
2. Loss is computed
3. Gradients are calculated using backpropagation
4. Optimizer updates trainable model parameters

The pipeline continuously minimizes training loss across epochs to improve emotional classification performance.

---

##### 📈 Validation Monitoring

After each epoch, the model is evaluated on the validation dataset.

The training pipeline tracks:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

These metrics help analyze:

- Model convergence
- Generalization capability
- Overfitting behavior

---

##### 💾 Best Model Checkpoint Saving

The pipeline automatically saves the best-performing model weights based on validation accuracy.

Saved file:

```bash
best_speech_model.pth
```

Storage location:

```bash
IIIT_project/
└── best_speech_model.pth
```

This checkpoint stores the learned parameters of the Speech-Only architecture and is later used during testing and inference.

---

##### 📊 Learning Curve Generation

During training, learning curves are automatically generated for:

- Loss Profiles
- Accuracy Profiles

Generated output:

```bash
IIIT_project/
└── Speech_only/
    └── Results/
        └── plots/
            └── Speech_model/
                └── learning_curves.png
```

The generated plots visualize training dynamics, convergence behavior, and validation performance across epochs.

<p align="center">
  <img src="assets/speech-learning-curves.png" width="80%">
</p>

---

##### ✅ Final Outcome

The Speech-Only training pipeline learns robust emotional speech representations by combining:

- Contextual speech embeddings
- Handcrafted acoustic features
- Temporal sequence modeling

to perform high-accuracy speech emotion recognition.
