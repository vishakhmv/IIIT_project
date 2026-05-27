# 🎭 Multimodal Emotion Recognition (Speech and Text)

## 📖 Project Overview

This project implements a **Multimodal Emotion Recognition System** capable of predicting human emotions using **speech-only, text-only, and combined speech-text inputs**.

The system classifies emotions into seven categories: **Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, and Sad**.

The repository contains three independent deep learning pipelines:

- 🎙️ **Speech-Only Model** — learns emotional patterns from acoustic speech features such as tone, pitch, and prosody.
- 📝 **Text-Only Model** — analyzes emotional meaning from the semantic and contextual understanding of transcribed text.
- 🔀 **Multimodal Fusion Model** — combines both speech and text representations to achieve stronger and more robust emotion recognition.

The project also includes **representation learning analysis, t-SNE visualization, confusion matrices, and error analysis** to study the effectiveness of multimodal learning.
📄 **Read the full Technical Report:** [Report.pdf](./Report.pdf)

---

## 📑 Table of Contents

- [🧠 Model Overview](#model-overview)
- [📊 Dataset](#dataset)
- [📦 Data Extraction & Splitting](#data-extraction-and-splitting)

- [🧠 Models](#models)
  - [🎙️ 1. Speech-Only Model](#speech-only-model)
  - [📝 2. Text-Only Model](#text-only-model)
  - [🔀 3. Multimodal Fusion Model](#fusion-model)

- [📈 Final Results Summary](#evaluation-summary)
- [⚠️ System Limitations & Constraints](#limitations)

- [⚙️ Installation & Usage](#installation-and-usage)
  - [📥 1. Download Requirements](#download-requirements)
  - [📁 2. Project Structure](#project-structure)
  - [📖 3. File & Folder Explanation](#file-and-folder-explanation)
  - [⬇️ 4. Clone Repository](#clone-repo)
  - [📂 5. After Cloning](#after-cloning)
  - [🛠️ 6. Install Dependencies](#install-dependencies)
  - [🚀 7. Run Training](#training)
  - [🧪 8. Run Testing](#testing)
  - [💬 9. Standalone MELD Text Emotion Recognition Model](#meld-text-model)
  - [🌐 10. Run Web Application](#web-app)

---

<a id="model-overview"></a>

## 🧠 Model Overview

| Model             | Input Modality | Main Backbone                       |
| ----------------- | -------------- | ----------------------------------- |
| Speech-Only       | Audio          | HuBERT + MFCC + BiLSTM              |
| Text-Only         | Text           | DistilBERT                          |
| Multimodal Fusion | Audio + Text   | HuBERT + MFCC + BiLSTM + DistilBERT |

<a id="dataset"></a>

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

> _“Say the word door”_  
> _“Say the word neat”_

All emotional categories contain the same spoken sentences, ensuring that emotional variation primarily comes from vocal expression rather than changes in textual content.

<a id="data-extraction-and-splitting"></a>

## 📦 Data Extraction & Splitting

The TESS dataset was extracted and processed using custom preprocessing scripts developed in Google Colab.

<p align="center">
  <img src="assets/tess-data-flow.png" width="85%">
</p>

---

<a id="data-extraction"></a>

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

<a id="data-splitting"></a>

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

| Audio File         | Generated Text    |
| ------------------ | ----------------- |
| YAF_back_angry.wav | say the word back |
| YAF_door_happy.wav | say the word door |

---

#### 🏷️ Emotion Label Encoding

Each emotion category was mapped into a numerical label for model training.

| Emotion           | Label |
| ----------------- | ----- |
| Angry             | 0     |
| Disgust           | 1     |
| Fear              | 2     |
| Happy             | 3     |
| Neutral           | 4     |
| Pleasant Surprise | 5     |
| Sad               | 6     |

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

| Column        | Description                   |
| ------------- | ----------------------------- |
| path          | Relative path to audio sample |
| text          | Generated text transcript     |
| label_encoded | Numerical emotion label       |

---

#### 📌 Example CSV Entry

| path                         | text              | label_encoded |
| ---------------------------- | ----------------- | ------------- |
| YAF_angry/YAF_back_angry.wav | say the word back | 0             |

---

#### ✅ Outcome

The final preprocessing pipeline produced a clean and reproducible multimodal dataset structure suitable for:

- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion Training

<a id="models"></a>

## 🧠 Models

This project consists of three independent deep learning pipelines:

---

<a id="speech-only-model"></a>

### 🎙️ 1. Speech-Only Model

The Speech-Only pipeline predicts emotions directly from raw audio signals by learning acoustic speech features such as tone, pitch, energy, etc.

---

#### 🏗️ a. System Architecture

The Speech-Only architecture predicts emotions directly from raw speech audio by combining pretrained contextual speech embeddings, handcrafted acoustic features, and temporal sequence modeling.

<p align="center">
  <img src="assets/speech-model-architecture.png" width="90%">
</p>

---

##### 🔊 1. Preprocessing

The model receives raw speech audio samples as input.

Each audio sample undergoes multiple preprocessing operations before feature extraction:

- Audio loading using `librosa`
- Silence trimming
- Fixed-length padding/truncation
- Sample rate normalization

To ensure consistent batch processing, all audio samples are standardized to a fixed duration and sampling rate.

---

##### 🎼 2. Feature Extraction

The architecture extracts two complementary speech representations simultaneously.

##### a. HuBERT Contextual Speech Embeddings

The raw waveform is passed through a pretrained **HuBERT (`facebook/hubert-base-ls960`)** model.

HuBERT generates high-dimensional contextual speech embeddings that capture:

- Speech context
- Prosody
- Temporal speech structure
- High-level acoustic semantics
- Phonetic and emotional speech patterns

The pretrained HuBERT weights are frozen during training to preserve learned speech representations and reduce computational overhead.

---

##### b. MFCC Acoustic Features

In parallel, MFCC (Mel-Frequency Cepstral Coefficient) features are extracted using `librosa`.

MFCC features capture low-level acoustic characteristics such as:

- Pitch
- Tone
- Frequency distribution
- Spectral variations
- Vocal tract characteristics

These handcrafted acoustic descriptors complement the contextual HuBERT representations.

---

##### c. Feature Fusion

The HuBERT embeddings and MFCC acoustic features are concatenated together along the feature dimension to create a unified speech representation.

This fusion mechanism enables the architecture to simultaneously learn from:

- Deep contextual speech representations
- Traditional handcrafted acoustic features

The fused representation provides richer emotional information compared to using either representation independently.

---

##### ⏳ 3. Temporal Modelling

The combined speech representation is passed through a **Bidirectional Long Short-Term Memory (BiLSTM)** network.

The BiLSTM models sequential emotional dependencies across speech frames and learns temporal speech dynamics such as:

- Emotional transitions
- Speaking rhythm
- Temporal prosodic variations
- Sequential acoustic patterns

Bidirectional processing allows the network to capture contextual dependencies from both forward and backward directions in the speech sequence.

---

##### 🌐 4. Temporal Pooling

After BiLSTM processing, global temporal pooling is applied across all time steps.

This operation compresses the sequential BiLSTM outputs into a compact fixed-dimensional emotional embedding representing the overall emotional characteristics of the speech sample.

---

##### 🎯 5. Classification

The pooled emotional embedding is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier predicts one of the following seven emotional categories:

| Emotion           | Label |
| ----------------- | ----- |
| Angry             | 0     |
| Disgust           | 1     |
| Fear              | 2     |
| Happy             | 3     |
| Neutral           | 4     |
| Pleasant Surprise | 5     |
| Sad               | 6     |

---

##### 📤 6. Final Emotion Prediction

The final output of the architecture is a probability distribution across all emotion classes.

The emotion with the highest probability score is selected as the predicted emotional state of the input speech sample.

---

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

| Column        | Description              |
| ------------- | ------------------------ |
| path          | Relative audio file path |
| text          | Generated transcript     |
| label_encoded | Numerical emotion label  |

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

| Parameter     | Value  |
| ------------- | ------ |
| Learning Rate | `1e-4` |
| Weight Decay  | `1e-2` |
| Batch Size    | `32`   |
| Epochs        | `10`   |

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
└── Results/
    └── plots/
        └── Speech_model/
            └── learning_curves.png
```

The generated plots visualize training convergence, validation stability, and overall learning behavior across training epochs.

<p align="center">
  <img src="assets/speech-learning-curve.png" width="80%">
</p>

The learning curves show rapid convergence with consistently low validation loss and high validation accuracy, indicating stable training and strong generalization performance on unseen validation samples.

---

##### ✅ Final Outcome

The Speech-Only training pipeline learns robust emotional speech representations by combining:

- Contextual speech embeddings
- Handcrafted acoustic features
- Temporal sequence modeling

to perform high-accuracy speech emotion recognition.

---

#### 📊 c. Testing & Evaluation Pipeline

The Speech-Only evaluation pipeline measures the model’s ability to generalize on unseen speech samples using multiple quantitative and visualization-based evaluation techniques.

The testing workflow includes:

- Test dataset loading
- Model checkpoint loading
- Inference generation
- Accuracy metric computation
- Confusion matrix generation
- t-SNE latent space visualization
- Performance report export

---

##### 📂 Test Dataset Loading

Evaluation samples are loaded from:

```bash
Data_split/test_split.csv
```

The testing pipeline uses the same `TESSSpeechDataset` preprocessing pipeline used during training to ensure consistent feature generation and inference behavior.

The test loader processes samples in deterministic order using:

```python
DataLoader(..., shuffle=False)
```

to preserve stable evaluation consistency.

---

##### 💾 Loading Trained Weights

The trained Speech-Only model weights are loaded from:

```bash
best_speech_model.pth
```

Storage location:

```bash
IIIT_project/
└── best_speech_model.pth
```

The checkpoint stores the learned parameters obtained during validation-based training.

---

##### 🧠 Inference Pipeline

During evaluation:

1. Raw speech audio is loaded
2. HuBERT embeddings are extracted
3. MFCC acoustic features are generated
4. Both feature representations are fused
5. The fused representation passes through the BiLSTM network
6. Temporal pooling generates emotional embeddings
7. The classifier predicts final emotion probabilities

The predicted emotion corresponds to the class with the highest probability score.

---

##### 📈 Classification Metrics

The evaluation pipeline computes detailed classification metrics using:

```python
classification_report()
```

The generated metrics strictly reflect class-wise and averaged performance, including:

- Precision
- Recall
- F1-Score
- Support
- Macro Average
- Weighted Average

Generated output:

```bash
IIIT_project/
└── Results/
    └── speech_classification_report.csv
```

<p align="center">
  <img src="assets/speech-classification-report.png" width="75%">
</p>

The Speech-Only model achieved an overall test accuracy of approximately **99.64%** on unseen evaluation samples, demonstrating highly robust emotional speech classification performance.

---

##### 🔥 Confusion Matrix Analysis

A confusion matrix is generated using:

```python
confusion_matrix()
```

to visualize class-wise prediction performance and misclassification behavior.

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Speech_model/
            └── confusion_matrix.png
```

<p align="center">
  <img src="assets/speech-confusion-matrix.png" width="70%">
</p>

The confusion matrix shows near-perfect class separation across all seven emotional categories, with only minimal confusion observed between emotionally similar speech classes.

---

##### 🌌 t-SNE Latent Space Visualization

The learned BiLSTM emotional embeddings are projected into 2D space using:

```python
TSNE()
```

to visualize how effectively the model separates emotional speech representations in latent feature space.

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Speech_model/
            └── tsne.png
```

<p align="center">
  <img src="assets/speech-tsne.png" width="75%">
</p>

The t-SNE visualization demonstrates strong inter-class separability, where emotionally distinct speech samples form highly compact and well-separated clusters in latent space.

---

##### 📊 Evaluation Summary

The Speech-Only evaluation pipeline demonstrates that the architecture successfully learns highly discriminative emotional speech representations using:

- Contextual HuBERT embeddings
- MFCC acoustic features
- Temporal BiLSTM sequence modeling

The near-perfect classification metrics and clearly separated latent clusters indicate strong generalization capability on unseen emotional speech samples.

---

<a id="text-only-model"></a>

### 📝 2. Text-Only Model

The Text-Only pipeline predicts emotions using text.

It learns semantic and contextual language representations directly from textual input using a pretrained transformer-based language model.

---

#### 🏗️ a. System Architecture

The Text-Only architecture performs emotion classification using pretrained transformer-based contextual language embeddings generated by DistilBERT.

<p align="center">
  <img src="assets/text-model-architecture.png" width="90%">
</p>

---

##### 📝 1. Preprocessing

The model receives textual input sequences representing speech transcripts.

Example:

```text
"say the word back"
```

The text preprocessing pipeline performs:

- Text tokenization
- Attention mask generation
- Sequence padding
- Sequence truncation

All text sequences are converted into fixed-length transformer-compatible representations for consistent batch processing.

---

##### 🔤 2. Feature Extraction

The input text is tokenized using the pretrained:

```python
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```

tokenizer.

The tokenizer converts raw text into transformer-compatible representations including:

- Input Token IDs
- Attention Masks
- Special Tokens (`[CLS]`, `[SEP]`)

These tokenized representations serve as inputs to the transformer encoder.

---

##### 🧠 3. Contextual Modelling

The tokenized text is passed through a pretrained:

```python
DistilBertModel('distilbert-base-uncased')
```

transformer language model.

DistilBERT generates high-dimensional contextual embeddings that capture:

- Linguistic structure
- Word relationships
- Contextual semantics
- Transformer attention dependencies
- Sentence-level semantic representations

The pretrained DistilBERT parameters are frozen during training to preserve learned language knowledge and reduce computational complexity.

---

##### 🌐 4. CLS Sentence Representation

After transformer encoding, the architecture extracts the contextual embedding corresponding to the `[CLS]` token:

```python
outputs.last_hidden_state[:, 0, :]
```

The CLS embedding acts as a compact sentence-level representation summarizing the semantic meaning of the entire input sequence.

---

##### 🎯 5. Classification

The CLS embedding is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier predicts one of the following seven emotional categories:

| Emotion           | Label |
| ----------------- | ----- |
| Angry             | 0     |
| Disgust           | 1     |
| Fear              | 2     |
| Happy             | 3     |
| Neutral           | 4     |
| Pleasant Surprise | 5     |
| Sad               | 6     |

---

##### 📤 6. Final Emotion Prediction

The final output of the architecture is a probability distribution across all emotional categories.

The emotion with the highest probability score is selected as the predicted emotional state of the input text sequence.

---

#### 🏋️ b. Training Pipeline

The Text-Only training pipeline learns emotional representations directly from textual input using pretrained transformer-based contextual embeddings generated by DistilBERT.

---

##### 📂 Dataset Loading

Training samples are loaded from:

```bash
Data_split/train_split.csv
```

The CSV file contains:

| Column        | Description              |
| ------------- | ------------------------ |
| path          | Relative audio file path |
| text          | Generated transcript     |
| label_encoded | Numerical emotion label  |

The complete training dataset is further divided into:

- Training Set
- Validation Set

using Scikit-learn’s:

```python
train_test_split()
```

utility with:

- **15% validation split**
- **Stratified class balancing**
- **Random state = 42**

This ensures balanced emotional category distributions during training and validation.

---

##### 📝 Text Preprocessing

The textual transcripts are processed using the pretrained:

```python
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```

tokenizer.

Each text sample undergoes:

- Tokenization
- Attention mask generation
- Sequence padding
- Sequence truncation

All sequences are converted into fixed-length transformer-compatible inputs before training.

---

##### 📦 DataLoader Pipeline

The processed text samples are loaded using custom:

```python
TESSTextDataset
```

dataset and PyTorch `DataLoader` pipelines.

Training configuration:

```python
batch_size = 32
shuffle = True
```

Validation configuration:

```python
batch_size = 32
shuffle = False
```

The shuffled training loader improves generalization while deterministic validation loading ensures stable evaluation consistency.

---

##### 🧠 Transformer Feature Extraction

The architecture uses pretrained:

```python
DistilBertModel('distilbert-base-uncased')
```

contextual language embeddings for semantic feature extraction.

During training:

- Transformer parameters are frozen
- Only the classification head is optimized

This significantly reduces computational cost while preserving pretrained language knowledge.

The extracted CLS embeddings act as compact sentence-level semantic representations for emotion classification.

---

##### 🔗 Classification Head Training

The CLS semantic embeddings are passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier learns mappings between semantic text representations and emotional categories.

---

##### ⚙️ Optimization Strategy

The training pipeline uses:

- **AdamW Optimizer**
- **CrossEntropy Loss**
- **Weight Decay Regularization**
- **Validation-based Checkpoint Saving**

Optimization configuration:

```python
learning_rate = 5e-4
weight_decay = 1e-2
epochs = 15
```

Only the classifier parameters are updated during optimization while DistilBERT remains frozen.

---

##### 💾 Best Model Checkpoint Saving

The best-performing model weights are automatically saved based on validation accuracy improvements.

Generated output:

```bash
IIIT_project/
└── best_text_model.pth
```

This checkpoint stores the learned parameters of the Text-Only emotion classification model and is later loaded during evaluation and inference.

---

##### 📈 Training Monitoring

During training, the pipeline continuously tracks:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

These metrics are stored epoch-wise for convergence analysis and performance monitoring.

---

##### 📊 Learning Curve Generation

Learning curves are automatically generated to visualize:

- Loss Profiles
- Accuracy Profiles
- Validation Stability
- Training Convergence Behavior

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Text_model/
            └── learning_curves.png
```

<p align="center">
  <img src="assets/text-learning-curve.png" width="80%">
</p>

The learning curves demonstrate extremely limited convergence and consistently low validation accuracy, highlighting the difficulty of performing emotion recognition using only semantic textual information from the TESS dataset.

This behavior occurs because the TESS dataset uses nearly identical carrier phrases across all emotional categories, providing minimal emotion-specific semantic variation for the transformer model to learn.

---

#### 📊 c. Testing & Evaluation Pipeline

The Text-Only evaluation pipeline measures the model’s ability to generalize on unseen textual inputs using quantitative metrics and latent space visualization techniques.

The evaluation workflow includes:

- Test dataset loading
- Model checkpoint loading
- Transformer inference
- Accuracy metric computation
- Confusion matrix generation
- t-SNE latent space visualization

---

##### 📂 Test Dataset Loading

Evaluation samples are loaded from:

```bash
Data_split/test_split.csv
```

using:

```python
pd.read_csv()
```

The testing pipeline uses the same `TESSTextDataset` preprocessing pipeline used during training to ensure consistent tokenization and transformer input generation.

The test loader processes samples using:

```python
DataLoader(..., shuffle=False)
```

to preserve deterministic evaluation consistency.

---

##### 💾 Loading Trained Weights

The trained Text-Only model weights are loaded from:

```bash
best_text_model.pth
```

Storage location:

```bash
IIIT_project/
└── best_text_model.pth
```

The checkpoint stores the learned parameters of the Text-Only classification model obtained during validation-based training.

---

##### 🧠 Transformer Inference Pipeline

During evaluation:

1. Input text is tokenized
2. Attention masks are generated
3. DistilBERT contextual embeddings are extracted
4. CLS token embeddings are generated
5. The classification head predicts emotional probabilities

The predicted emotion corresponds to the class with the highest probability score.

---

##### 📈 Classification Metrics

The evaluation pipeline computes detailed classification metrics using:

```python
classification_report()
```

The generated metrics strictly reflect class-wise and averaged performance, including:

- Precision
- Recall
- F1-Score
- Support
- Macro Average
- Weighted Average

Generated output:

```bash
IIIT_project/
└── Results/
    └── text_classification_report.csv
```

<p align="center">
  <img src="assets/text-classification-report.png" width="75%">
</p>

The Text-Only model achieved an overall test accuracy of approximately **14.29%**, with extremely low precision, recall, and F1-scores across most emotional categories.

The classification report shows that the model predominantly predicted a single emotion class (“Pleasant Surprise”), resulting in near-zero performance for the remaining categories. This behavior indicates poor class discrimination and highlights the limitations of relying solely on textual semantic information for emotion recognition on the TESS dataset.

---

##### 🔥 Confusion Matrix Analysis

A confusion matrix is generated using:

```python
confusion_matrix()
```

to visualize class-wise prediction behavior and semantic confusion patterns.

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Text_model/
            └── confusion_matrix.png
```

<p align="center">
  <img src="assets/text-confusion-matrix.png" width="70%">
</p>

The confusion matrix shows severe prediction collapse, where the model predicts nearly all samples as a single emotional category.

This indicates that the transformer model was unable to learn discriminative semantic representations for different emotions.

---

##### 🌌 t-SNE Latent Space Visualization

The learned DistilBERT semantic embeddings are projected into 2D space using:

```python
TSNE()
```

to visualize latent feature separability across emotional categories.

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Text_model/
            └── tsne.png
```

<p align="center">
  <img src="assets/text-tsne.png" width="75%">
</p>

The t-SNE visualization shows highly overlapping emotional representations with no clearly separable clusters.

Unlike the Speech-Only model, the semantic embeddings fail to organize into meaningful emotional groupings in latent space.

---

##### ⚠️ Failure Analysis

The Text-Only model demonstrated extremely poor generalization performance on the TESS dataset because the dataset provides very limited emotion-specific semantic information.

The TESS dataset uses identical carrier phrases across all emotional categories.

For example, the sentence:

```text
"say the word back"
```

appears in all seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise
- Sad

with only the vocal delivery changing.

As a result:

- The textual modality contains minimal emotional variance
- DistilBERT cannot learn meaningful semantic emotion separation
- The model collapses toward dominant prediction behavior
- Latent representations fail to form separable emotional clusters

This behavior demonstrates a major limitation of standalone semantic emotion recognition and highlights the critical importance of acoustic speech information for understanding human emotions in the TESS dataset.

---

##### 📊 Evaluation Summary

The Text-Only evaluation pipeline demonstrates that transformer-based semantic representations alone are insufficient for robust emotion recognition on linguistically constrained datasets such as TESS.

The failure of semantic-only learning motivates the need for:

- Acoustic speech representations
- Prosodic information
- Multimodal feature fusion

for effective emotional understanding.

---

#### 💬 d. Experimental Extension — Standalone MELD Text Emotion Recognition

A standalone **RoBERTa-based** text emotion recognition model was experimentally trained and evaluated on the **MELD (Multimodal EmotionLines Dataset)** conversational emotion corpus to investigate transformer-based emotion understanding in dialogue contexts. Unlike TESS, MELD provides multi-speaker conversational dialogue with richer contextual and emotional variations, enabling transformer architectures to leverage semantic representations more effectively.

> ⚠️ Since TESS and MELD differ in domain, size, and class distribution, performance differences reflect dataset characteristics as much as architectural suitability and should not be directly compared.

---

##### 🏗️ Architecture & Tools

The standalone MELD model is built around a **RoBERTa (`roberta-base`)** transformer encoder fine-tuned for 7-class emotion classification.

| Component | Details |
|-----------|---------|
| **Base Model** | `roberta-base` (HuggingFace Transformers) |
| **Tokenizer** | `RobertaTokenizer` — max sequence length 128 |
| **Classification Head** | Linear layer over `[CLS]` token embedding → 7 emotion classes |
| **Optimizer** | AdamW — lr `2e-5` |
| **Loss** | CrossEntropyLoss |
| **Batch Size** | 16 |
| **Epochs** | 5 |
| **Framework** | PyTorch |

Unlike the TESS text model (which freezes DistilBERT and trains only the classifier head), the **entire RoBERTa model is fine-tuned end-to-end** here, allowing it to adapt its pretrained language representations to the emotion-labeled conversational dialogue domain.

---

##### 📊 MELD Classification Performance

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| neutral  | 0.74      | 0.86   | 0.80     | 1256    |
| surprise | 0.58      | 0.53   | 0.55     | 281     |
| fear     | 0.21      | 0.06   | 0.09     | 50      |
| sadness  | 0.46      | 0.25   | 0.33     | 208     |
| joy      | 0.60      | 0.60   | 0.60     | 402     |
| disgust  | 0.48      | 0.18   | 0.26     | 68      |
| anger    | 0.48      | 0.47   | 0.48     | 345     |

---

##### 📈 Overall Performance

| Metric            | Score  |
|-------------------|--------|
| Accuracy          | 65.25% |
| Macro F1-Score    | 0.44   |
| Weighted F1-Score | 0.63   |

---

##### 🖼️ Evaluation Visualizations

<p align="center">
  <img src="assets/mcf.png" width="48%" />
  <img src="assets/mtsne.png" width="48%" />
</p>
<p align="center">
  <em>Left: Confusion Matrix &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Right: t-SNE Embedding Visualization</em>
</p>

---

The model performs reliably on dominant classes such as **neutral** (F1 = 0.80) and **joy** (F1 = 0.60), but struggles with minority emotions — particularly **fear** (F1 = 0.09) and **disgust** (F1 = 0.26) — due to limited training samples and semantic overlap with related classes. The notable gap between accuracy (65.25%) and macro F1 (0.44) further reflects the dataset's class imbalance. These findings highlight the importance of multimodal fusion — combining textual, acoustic, and visual signals — for more robust and generalised emotion recognition.

---

<a id="fusion-model"></a>

### 🔀 3. Multimodal Fusion Model

The Multimodal Fusion pipeline predicts emotions by jointly learning from both speech and textual representations.

Unlike the standalone Speech-Only and Text-Only architectures, this model combines acoustic speech features with contextual semantic language embeddings to create a unified multimodal emotional representation.

The fusion architecture integrates:

- Contextual speech embeddings from HuBERT
- MFCC acoustic speech features
- Transformer-based semantic text embeddings from DistilBERT

to perform robust multimodal emotion recognition.

---

#### 🏗️ a. System Architecture

The Multimodal Fusion architecture combines parallel speech and text processing branches into a unified multimodal emotional representation for final emotion classification.

<p align="center">
  <img src="assets/fusion-model-architecture.png" width="95%">
</p>

---

##### 🎙️ 1. Speech Branch

The speech branch learns emotional information directly from raw audio signals using contextual speech embeddings, handcrafted acoustic features, and temporal sequence modelling.

---

##### 🔊 1.1 Speech Preprocessing

The model receives raw speech audio samples as input.

Each audio sample undergoes multiple preprocessing operations including:

- Audio loading using `librosa`
- Silence trimming
- Fixed-length padding/truncation
- Sample rate normalization

All speech samples are standardized to a fixed duration and sampling rate to ensure consistent batch processing.

---

##### 🎼 1.2 Speech Feature Extraction

The architecture extracts two complementary speech representations simultaneously.

##### a. HuBERT Contextual Speech Embeddings

Raw speech waveforms are passed through the pretrained:

```python
HubertModel.from_pretrained("facebook/hubert-base-ls960")
```

model.

HuBERT generates contextual speech embeddings that capture:

- Prosody
- Temporal speech structure
- Acoustic context
- Emotional speech dynamics
- High-level speech semantics

The pretrained HuBERT parameters are frozen during training to preserve learned speech representations and reduce computational complexity.

---

##### b. MFCC Acoustic Features

In parallel, handcrafted MFCC (Mel-Frequency Cepstral Coefficient) features are extracted using `librosa`.

MFCC features capture low-level acoustic characteristics including:

- Pitch
- Tone
- Spectral structure
- Frequency distribution
- Vocal tract variations

These handcrafted descriptors complement the contextual HuBERT embeddings.

---

##### ⏳ 1.3 Speech Temporal Modelling

The HuBERT embeddings and MFCC features are concatenated together and passed through a Bidirectional Long Short-Term Memory (BiLSTM) network.

The BiLSTM learns temporal emotional dependencies across speech sequences including:

- Emotional transitions
- Speaking rhythm
- Prosodic variations
- Sequential acoustic patterns

Bidirectional processing allows the architecture to capture emotional information from both forward and backward temporal directions.

Global temporal pooling is then applied to generate a compact speech emotional embedding.

---

##### 📝 2. Text Branch

The text branch learns semantic emotional information using pretrained transformer-based contextual language embeddings.

---

##### 📝 2.1 Text Preprocessing

The model receives textual input sequences representing speech transcripts.

Example:

```text
"say the word back"
```

The text preprocessing pipeline performs:

- Text tokenization
- Attention mask generation
- Sequence padding
- Sequence truncation

All text sequences are converted into fixed-length transformer-compatible representations for consistent batch processing.

---

##### 🔤 2.2 Text Feature Extraction

The textual inputs are tokenized using the pretrained:

```python
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```

tokenizer.

The tokenizer converts raw text into transformer-compatible representations including:

- Input Token IDs
- Attention Masks
- Special Tokens (`[CLS]`, `[SEP]`)

These tokenized representations serve as inputs to the transformer encoder.

---

##### 🧠 2.3 Contextual Modelling

The tokenized text is processed using the pretrained:

```python
DistilBertModel.from_pretrained('distilbert-base-uncased')
```

transformer language model.

DistilBERT generates contextual semantic embeddings that capture:

- Linguistic structure
- Contextual word relationships
- Sentence-level semantics
- Transformer attention dependencies

The pretrained DistilBERT parameters are frozen during training to preserve pretrained language knowledge and reduce computational complexity.

After transformer encoding, the architecture extracts the contextual embedding corresponding to the `[CLS]` token:

```python
outputs.last_hidden_state[:, 0, :]
```

The CLS embedding acts as a compact sentence-level semantic representation summarizing the entire input sequence.

---

##### 🔗 3. Multimodal Fusion

The pooled speech embedding and DistilBERT semantic embedding are concatenated together to form a unified multimodal emotional representation.

This fusion mechanism enables the architecture to simultaneously learn from:

- Acoustic emotional cues
- Temporal speech dynamics
- Semantic contextual information

The fused representation contains richer emotional information than either modality independently.

---

##### 🎯 4. Classification

The fused multimodal representation is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier predicts one of the following seven emotional categories:

| Emotion           | Label |
| ----------------- | ----- |
| Angry             | 0     |
| Disgust           | 1     |
| Fear              | 2     |
| Happy             | 3     |
| Neutral           | 4     |
| Pleasant Surprise | 5     |
| Sad               | 6     |

---

##### 📤 5. Final Emotion Prediction

The final output of the Multimodal Fusion architecture is a probability distribution across all emotional categories.

The emotion with the highest probability score is selected as the predicted emotional state by jointly analyzing both speech and textual modalities.

---

#### 🏋️ b. Training Pipeline

The Multimodal Fusion training pipeline jointly learns emotional representations from both speech and textual modalities using supervised deep learning.

The architecture combines:

- Contextual speech embeddings
- Handcrafted acoustic speech features
- Transformer-based semantic language embeddings

to create a unified multimodal emotional representation for emotion classification.

---

##### 📂 Dataset Loading

Training samples are loaded from:

```bash
Data_split/train_split.csv
```

using:

```python
pd.read_csv()
```

The CSV file contains:

| Column        | Description              |
| ------------- | ------------------------ |
| path          | Relative audio file path |
| text          | Speech transcript        |
| label_encoded | Numerical emotion label  |

The multimodal training pipeline simultaneously uses:

- `path` → Speech modality input
- `text` → Text modality input
- `label_encoded` → Emotion supervision label

---

##### ✂️ Training–Validation Split

The complete training dataset is further divided into:

- Training Set
- Validation Set

using Scikit-learn’s:

```python
train_test_split()
```

utility with:

- **15% validation split**
- **Stratified class balancing**
- **Random state = 42**

This preserves balanced emotional distributions across both subsets.

The validation dataset is used to:

- Monitor generalization performance
- Track validation accuracy
- Prevent overfitting
- Save the best-performing model checkpoint

---

##### 🎙️ Speech Preprocessing

Each speech sample undergoes multiple preprocessing operations before feature extraction.

The speech preprocessing pipeline performs:

- Audio loading using `librosa`
- Silence trimming
- Fixed-length padding/truncation
- Sample rate normalization
- MFCC feature extraction

Audio samples are standardized to fixed-duration waveforms for stable multimodal batch processing.

---

##### 📝 Text Preprocessing

The textual transcripts are processed using the pretrained:

```python
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```

tokenizer.

Each text sample undergoes:

- Tokenization
- Attention mask generation
- Sequence padding
- Sequence truncation

All text sequences are converted into fixed-length transformer-compatible representations before training.

---

##### 📦 Multimodal Dataset Pipeline

The processed speech and text samples are loaded using the custom:

```python
TESSMultimodalDataset
```

dataset and PyTorch `DataLoader` pipelines.

Each training sample simultaneously contains:

- Raw speech waveform
- MFCC acoustic features
- Tokenized text embeddings
- Attention masks
- Emotion labels

Training configuration:

```python
batch_size = 32
shuffle = True
```

Validation configuration:

```python
batch_size = 32
shuffle = False
```

The shuffled training loader improves generalization, while deterministic validation loading ensures stable evaluation consistency.

---

##### 🧠 Parallel Feature Extraction

During training, the architecture extracts multimodal representations using parallel speech and text processing branches.

---

##### 1. Speech Feature Extraction

The speech branch extracts:

- Contextual HuBERT embeddings
- MFCC acoustic features

The pretrained:

```python
HubertModel.from_pretrained("facebook/hubert-base-ls960")
```

model generates contextual speech embeddings directly from raw audio waveforms.

The HuBERT parameters remain frozen during training to preserve pretrained speech knowledge and reduce computational overhead.

---

##### 2. Text Feature Extraction

The text branch extracts contextual semantic embeddings using pretrained DistilBERT.

The pretrained:

```python
DistilBertModel.from_pretrained('distilbert-base-uncased')
```

model generates transformer-based semantic language representations from textual input.

The `[CLS]` token embedding acts as the sentence-level semantic representation.

The DistilBERT parameters remain frozen during training to preserve pretrained language knowledge.

---

##### ⏳ Speech Temporal Modeling

After extracting the HuBERT embeddings and MFCC features, the combined speech representations are processed through a Bidirectional Long Short-Term Memory (BiLSTM) network.

The BiLSTM learns:

- Emotional transitions
- Sequential speech dynamics
- Temporal prosodic patterns
- Contextual acoustic dependencies

Bidirectional processing allows the network to capture emotional information from both forward and backward temporal directions.

Global temporal pooling is applied after BiLSTM processing to generate a compact, fixed-dimensional emotional speech embedding.

---

##### 🔗 Multimodal Feature Fusion (Late Fusion)

The pooled speech representation (from the BiLSTM) and the DistilBERT semantic embedding (the `[CLS]` token) are concatenated together along the feature dimension to create a unified multimodal emotional representation.

This fusion mechanism enables the architecture to jointly learn from:

- Acoustic emotional cues
- Temporal speech dynamics
- Semantic contextual information

The fused representation contains richer emotional information than either modality independently, serving as the ultimate joint-feature vector for the network.

---

##### 🎯 Emotion Classification

The final fused multimodal representation is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classifier predicts one of the seven emotional categories.

---

##### 📉 Loss Function

The training pipeline uses:

```python
CrossEntropyLoss()
```

to measure prediction error between:

- Predicted emotion probabilities
- Ground-truth emotion labels

This loss function is suitable for multi-class multimodal emotion classification tasks.

---

##### ⚙️ Optimization Strategy

The multimodal fusion model parameters are optimized using:

```python
torch.optim.AdamW()
```

Optimization configuration:

| Parameter     | Value  |
| ------------- | ------ |
| Learning Rate | `1e-4` |
| Weight Decay  | `1e-2` |
| Batch Size    | `32`   |
| Epochs        | `10`   |

The optimizer updates trainable fusion and classification parameters using gradient-based backpropagation.

---

##### 🔄 Backpropagation & Parameter Updates

For each training batch:

1. Speech and text features are extracted
2. Speech temporal embeddings are generated
3. Semantic text embeddings are extracted
4. Multimodal features are fused
5. Emotion probabilities are predicted
6. Loss is computed
7. Gradients are calculated
8. Model parameters are updated

The pipeline continuously minimizes training loss across epochs to improve multimodal emotional understanding.

---

##### 📈 Validation Monitoring

After each epoch, the model is evaluated on the validation dataset.

The training pipeline tracks:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

These metrics help analyze:

- Fusion learning stability
- Model convergence
- Generalization capability
- Overfitting behavior

---

##### 💾 Best Model Checkpoint Saving

The best-performing multimodal fusion model weights are automatically saved based on validation accuracy improvements.

Generated output:

```bash
IIIT_project/
└── best_fusion_model.pth
```

This checkpoint stores the learned parameters of the Multimodal Fusion architecture and is later used during testing and inference.

---

##### 📊 Learning Curve Generation

Learning curves are automatically generated to visualize:

- Loss Profiles
- Accuracy Profiles
- Validation Stability
- Fusion Learning Convergence

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Fusion_model/
            └── learning_curves.png
```

<p align="center">
  <img src="assets/fusion-learning-curve.png" width="80%">
</p>

The learning curves demonstrate rapid convergence with consistently low validation loss and near-perfect validation accuracy.

Compared to the standalone Text-Only model, the Multimodal Fusion architecture learns significantly more stable and discriminative emotional representations by combining:

- Acoustic speech information
- Temporal speech dynamics
- Semantic contextual language embeddings

The close alignment between training and validation curves also indicates strong generalization performance with minimal overfitting.

---

##### ✅ Final Outcome

The Multimodal Fusion training pipeline successfully learns complementary emotional representations from both speech and text modalities.

By combining:

- Contextual speech embeddings
- Handcrafted acoustic features
- Temporal sequence modeling
- Transformer-based semantic embeddings

the architecture achieves highly robust multimodal emotion recognition performance compared to standalone unimodal models.

---

#### 📊 c. Testing & Evaluation Pipeline

The Multimodal Fusion evaluation pipeline measures the model’s ability to generalize on unseen multimodal samples using quantitative metrics and latent-space visualization techniques.

The evaluation workflow includes:

- Test dataset loading
- Model checkpoint loading
- Multimodal inference generation
- Accuracy metric computation
- Confusion matrix generation
- t-SNE latent space visualization
- Performance report export

---

##### 📂 Test Dataset Loading

Evaluation samples are loaded from:

```bash
Data_split/test_split.csv
```

using:

```python
pd.read_csv()
```

The testing pipeline uses the same:

```python
TESSMultimodalDataset
```

preprocessing pipeline used during training to ensure consistent speech and text feature generation.

Each evaluation sample contains:

- Raw speech waveform
- MFCC acoustic features
- Tokenized text embeddings
- Attention masks
- Emotion labels

The test loader processes samples using:

```python
DataLoader(..., shuffle=False)
```

to preserve deterministic evaluation consistency.

---

##### 💾 Loading Trained Weights

The trained Multimodal Fusion model weights are loaded from:

```bash
best_fusion_model.pth
```

Storage location:

```bash
IIIT_project/
└── best_fusion_model.pth
```

The checkpoint stores the learned parameters of the multimodal fusion architecture obtained during validation-based training.

---

##### 🧠 Multimodal Inference Pipeline

During evaluation:

1. Raw speech audio is loaded
2. HuBERT contextual speech embeddings are extracted
3. MFCC acoustic features are generated
4. Speech features are processed through the BiLSTM network
5. Temporal pooling generates compact speech embeddings
6. Text transcripts are tokenized using DistilBERT tokenizer
7. DistilBERT semantic embeddings are extracted
8. Speech and text embeddings are fused together
9. The classifier predicts final emotion probabilities

The predicted emotion corresponds to the class with the highest probability score.

---

##### 📈 Classification Metrics

The evaluation pipeline computes detailed classification metrics using:

```python
classification_report()
```

The generated metrics strictly reflect class-wise and averaged performance, including:

- Precision
- Recall
- F1-Score
- Support
- Macro Average
- Weighted Average

Generated output:

```bash
IIIT_project/
└── Results/
    └── fusion_classification_report.csv
```

<p align="center">
  <img src="assets/fusion-classification-report.png" width="75%">
</p>

The Multimodal Fusion model achieved an overall test accuracy of approximately **99.29%** on unseen evaluation samples.

The generated classification report demonstrates exceptionally strong performance across all emotional categories, with precision, recall, and F1-scores remaining consistently close to perfect values.

Only minor variations are observed in emotionally similar categories such as:

- Neutral
- Pleasant Surprise
- Disgust
- Sad

Despite these slight variations, the model maintains highly reliable and balanced classification performance across all seven emotion classes.

---

##### 🔥 Confusion Matrix Analysis

A confusion matrix is generated using:

```python
confusion_matrix()
```

to visualize class-wise prediction behavior and multimodal classification performance.

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Fusion_model/
            └── confusion_matrix.png
```

<p align="center">
  <img src="assets/fusion-confusion-matrix.png" width="70%">
</p>

The confusion matrix demonstrates near-perfect emotional class separation across all seven categories.

Most emotions are classified with 100% accuracy, while only a few minor misclassifications occur between emotionally related categories:

- Neutral → Sad
- Pleasant Surprise → Disgust

These small confusions indicate that the model learns highly discriminative multimodal emotional representations while still reflecting subtle emotional overlap between certain speech patterns.

Compared to the Text-Only model, the fusion architecture shows dramatically improved emotional discrimination due to the inclusion of rich acoustic speech information.

---

##### 🌌 t-SNE Latent Space Visualization

The learned multimodal embeddings are projected into 2D space using:

```python
TSNE()
```

to visualize latent feature separability across emotional categories.

Generated output:

```bash
IIIT_project/
└── Results/
    └── plots/
        └── Fusion_model/
            └── tsne.png
```

<p align="center">
  <img src="assets/fusion-tsne.png" width="75%">
</p>

The t-SNE visualization demonstrates highly compact and clearly separated emotional clusters in latent feature space.

Unlike the Text-Only model, where emotional embeddings heavily overlap, the fusion model organizes emotional representations into distinct regions with strong inter-class separation.

This indicates that the architecture successfully learns robust multimodal emotional representations by combining:

- Contextual speech embeddings
- Acoustic MFCC features
- Temporal speech dynamics
- Semantic language information

The strong clustering behavior further validates the effectiveness of multimodal fusion for emotion recognition.

---

##### 🧩 Why Fusion Works Despite Text-Only Failure

Although the Text-Only model performs poorly on the TESS dataset, the Multimodal Fusion architecture still achieves extremely high accuracy because the fusion model primarily relies on the highly informative acoustic speech modality.

In the TESS dataset:

- Speech signals contain strong emotional information through tone, pitch, prosody, and vocal dynamics
- Text transcripts contain limited semantic emotional variation

As a result:

- The speech branch dominates emotional discrimination
- The text branch acts as a supplementary contextual modality
- Fusion improves representation robustness rather than replacing speech understanding

The multimodal architecture therefore benefits from combining:

- Strong acoustic emotional cues
- Weak but complementary semantic information

This demonstrates an important principle in multimodal learning:

> Even weak modalities can improve representation stability when combined with highly informative modalities.

---

##### 📊 Evaluation Summary

The Multimodal Fusion evaluation pipeline demonstrates that combining speech and text modalities produces significantly stronger emotional representations than standalone unimodal approaches.

By integrating:

- Contextual HuBERT speech embeddings
- MFCC acoustic features
- Temporal BiLSTM sequence modeling
- DistilBERT semantic language embeddings

the architecture achieves highly robust multimodal emotion recognition performance with strong generalization capability on unseen emotional speech samples.

<a id="evaluation-summary"></a>

## 📈 Final Results Summary

The comparative evaluation of all three architectures demonstrates the importance of acoustic speech information for robust emotion recognition on the TESS dataset.

| Model             | Input Modality | Test Accuracy |
| ----------------- | -------------- | ------------- |
| Speech-Only       | Audio          | 99.64%        |
| Text-Only         | Text           | 14.29%        |
| Multimodal Fusion | Audio + Text   | 99.29%        |

---

### 📊 Key Observations

- The **Speech-Only Model** achieved the highest overall accuracy, demonstrating the effectiveness of contextual speech embeddings and temporal acoustic modeling for emotion recognition.

- The **Text-Only Model** performed poorly because the TESS dataset contains nearly identical carrier phrases across all emotional categories, providing minimal semantic emotional variation.

- The **Multimodal Fusion Model** achieved highly robust performance by combining:
  - contextual speech embeddings,
  - handcrafted acoustic features,
  - temporal speech dynamics,
  - and semantic language representations.

- Multimodal fusion enables the architecture to simultaneously learn from:
  - vocal emotion cues,
  - speech prosody,
  - acoustic frequency patterns,
  - and contextual semantic information.

- The fusion mechanism improves emotional representation learning by leveraging complementary strengths from both speech and text modalities, resulting in stronger generalization and more robust emotion classification.

- The fusion architecture successfully learned complementary multimodal emotional representations while overcoming the limitations of standalone text-based learning.

---

### ✅ Final Conclusion

This project demonstrates that acoustic speech information plays a dominant role in emotion recognition for linguistically constrained datasets such as TESS.

The experimental results further show that multimodal learning can effectively integrate speech and language representations to build highly accurate and robust emotion recognition systems.

---
---

<a id="limitations"></a>
## ⚠️ System Limitations & Constraints

While the Speech-Only and Multimodal Fusion models achieve highly robust accuracy on the testing data, both pipelines possess inherent limitations tied directly to their baseline training environment:

- **Gender and Age Bias:** The TESS dataset exclusively features recordings from two female actresses (aged 26 and 64). Because the underlying feature encoders have never been exposed to male voices, children's voices, or diverse vocal pitches outside this specific demographic, both the Speech-Only and Multimodal Fusion models will struggle to accurately generalize emotional states across a broader human population.
- **Environmental Noise Sensitivity:** All training samples consist of clean, studio-recorded audio. The acoustic feature extractors have not been exposed to raw waveform data augmentation involving environmental noise. As a result, both the speech-only and multimodal fusion systems may experience notable performance degradation in real-world settings containing ambient background noise, overlapping speech, or low-quality microphone inputs.
- **Textual Lexical Variance:** As highlighted in the failure analysis, the Text-Only model collapses on the TESS dataset due to the use of identical carrier phrases. While the fusion model successfully utilizes the audio modality to disambiguate and gate this out, the current semantic branch remains unequipped to handle complex, real-world conversational text without being structurally retrained on a more linguistically diverse dataset (e.g., AICHE, IEMOCAP, or MELD).

These limitations provide a clear pathway for **Future Work**, including training on diverse, multi-speaker datasets and applying noise-injection techniques to improve real-world robustness for all acoustic-based pipelines.

---

<a id="installation-and-usage"></a>

## ⚙️ Installation & Usage

<a id="download-requirements"></a>

### 📥 1. Download Requirements

Before running the project, download the required dataset and pretrained model checkpoints.

---

#### 📊 TESS Dataset Setup

Download the official TESS dataset in `.zip` format from Kaggle:

- [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

After downloading:

1. Keep the original compressed dataset as:

```bash
archive.zip
```

2. Extract the dataset folder:

```bash
TESS Toronto emotional speech set data/
```

3. Place both the `.zip` file and extracted dataset folder inside:

```bash
IIIT_project/
└── TESS_dataset/
    ├── archive.zip
    └── TESS Toronto emotional speech set data/
```

---

##### 📌 Why Both ZIP and Extracted Dataset Are Maintained

Both versions are intentionally maintained because different project components use them differently.

| File / Folder                             | Purpose                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| `archive.zip`                             | Used by `colab_data_extraction.py` during dataset extraction and preprocessing |
| `TESS Toronto emotional speech set data/` | Used directly by dataset split generation, training, and testing pipelines     |

The extraction pipeline first reads the compressed archive and extracts the dataset locally.

The dataset split generation script (`colab_data_split.py`) then scans the extracted dataset directory to create:

- `train_split.csv`
- `test_split.csv`

All model pipelines (`train.py` and `test.py`) directly load audio samples from the extracted dataset folder.

---

##### ⚠️ Important Note

If you are **only running the model training and testing pipelines** (`train.py` and `test.py`) using already generated dataset split CSV files, the `archive.zip` file is **not required**.

Only the extracted dataset folder is necessary in that case:

```bash
TESS_dataset/
└── TESS Toronto emotional speech set data/
```

The compressed `archive.zip` file is only required when running:

- `colab_data_extraction.py`
- `colab_data_split.py`

---

#### 📦 Pretrained Model Weights

Download pretrained model checkpoints from the following Google Drive links:

| Model                      | Download Link                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| 🎙️ Speech-Only Model       | [Download](https://drive.google.com/file/d/1ljNiWmN_klH4EFLnPHLLsIVzHvRtrQaA/view?usp=drive_link) |
| 📝 Text-Only Model         | [Download](https://drive.google.com/file/d/1CCUGR08Ozg_9bhWtyu4Ib6qasQiJ1nO8/view?usp=drive_link) |
| 🔀 Multimodal Fusion Model | [Download](https://drive.google.com/file/d/1oP1F4-bpNNY3zwPs2Q1W7l_kNx5ptRvT/view?usp=drive_link) |

All checkpoints are also available together in a single Drive folder:

- https://drive.google.com/drive/folders/1SWWakvZYEBRarcdlSpeEHXrHE2zwQO8N?usp=drive_link

After downloading, place the pretrained weights inside the root project directory:

```bash
IIIT_project/
├── best_speech_model.pth
├── best_text_model.pth
├── best_fusion_model.pth
```

---

<a id="project-structure"></a>

### 📁 2. Project Structure

```bash
IIIT_project/
│
├── assets/
│
├── Data_split/
│   ├── colab_data_extraction.py
│   ├── colab_data_split.py
│   ├── train_split.csv
│   └── test_split.csv
│
├── project/
│   ├── models/
│   │   ├── speech_pipeline/
│   │   │   ├── train.py
│   │   │   └── test.py
│   │   │
│   │   ├── text_pipeline/
│   │   │   ├── train.py
│   │   │   └── test.py
│   │   │
│   │   └── fusion_pipeline/
│   │       ├── train.py
│   │       └── test.py
│   │
│   ├── Results/
│   │   ├── plots/
│   │   │   ├── Speech_model/
│   │   │   ├── Text_model/
│   │   │   └── Fusion_model/
│   │   │
│   │   ├── speech_classification_report.csv
│   │   ├── text_classification_report.csv
│   │   ├── fusion_classification_report.csv
│   │   └── variant_accuracy_table.csv
│   │
│   └── requirements.txt
│
├── TESS_dataset/
│   ├── TESS Toronto emotional speech set data/
│   └── archive.zip
│
├── Web/
│   ├── app.py
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── requirements.txt
|
├── MELD_text_model/
│   ├── MELD_dataset/
│   │   ├── MELD-Features-Models/
│   │   └── MELD-RAW/
│   │
│   ├── train.py
│   ├── test.py
│   ├── requirements.txt
│   │
│   ├── best_meld_text_model.pth
│   ├── learning_curve.png
│   ├── confusion_matrix.png
│   ├── tsne_plot.png
│   └── classification_report.txt
│
├── best_speech_model.pth
├── best_text_model.pth
├── best_fusion_model.pth
│
├── .gitignore
├── README.md
└── Report.pdf

```

> **⚠️ Note:** The `MELD_text_model/` folder is an experimental extension and is entirely optional. It is only required if you specifically wish to train or evaluate the MELD Text Model. For full instructions, please see the [Standalone MELD Text Emotion Recognition Model](#meld-text-model).

---

<a id="file-and-folder-explanation"></a>

### 📖 3. File & Folder Explanation

| File / Folder                               | Description                                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `assets/`                                   | Contains README visualization assets, architecture diagrams, learning curves, confusion matrices, and t-SNE plots.  |
| `Data_split/`                               | Contains dataset preprocessing scripts and generated train-test split CSV files.                                    |
| `↳ colab_data_extraction.py`                | Extracts and prepares the original TESS dataset from the compressed archive.                                        |
| `↳ colab_data_split.py`                     | Generates structured training and testing CSV splits from the extracted dataset.                                    |
| `↳ train_split.csv`                         | Training metadata file containing audio paths, generated transcripts, and encoded emotion labels.                   |
| `↳ test_split.csv`                          | Testing metadata file containing audio paths, generated transcripts, and encoded emotion labels.                    |
| `project/models/`                           | Contains all Speech-Only, Text-Only, and Multimodal Fusion training and evaluation pipelines.                       |
| `↳ speech_pipeline/`                        | Training and evaluation pipeline for the Speech-Only architecture.                                                  |
| `↳ text_pipeline/`                          | Training and evaluation pipeline for the Text-Only architecture.                                                    |
| `↳ fusion_pipeline/`                        | Training and evaluation pipeline for the Multimodal Fusion architecture.                                            |
| `↳ train.py`                                | Trains the corresponding architecture.                                                                              |
| `↳ test.py`                                 | Evaluates the trained model and generates metrics and visualizations.                                               |
| `project/Results/`                          | Stores generated evaluation metrics, plots, and experimental outputs.                                               |
| `↳ plots/`                                  | Contains confusion matrices, learning curves, and t-SNE visualizations for all models.                              |
| `↳ speech_classification_report.csv`        | Class-wise precision, recall, and F1-score for the Speech model.                                                    |
| `↳ text_classification_report.csv`          | Class-wise precision, recall, and F1-score for the Text model.                                                      |
| `↳ fusion_classification_report.csv`        | Class-wise precision, recall, and F1-score for the Fusion model.                                                    |
| `↳ variant_accuracy_table.csv`              | Summary table comparing overall test accuracy across all three architectures.                                       |
| `project/requirements.txt`                  | Python dependencies required for model training, evaluation, and visualization pipelines.                           |
| `Web/`                                      | Web-based interface for real-time multimodal emotion recognition using the trained Speech, Text, and Fusion models. |
| `↳ app.py`                                  | Flask backend responsible for loading trained models and performing local emotion inference.                        |
| `↳ index.html`                              | Main frontend webpage for user interaction and emotion prediction.                                                  |
| `↳ style.css`                               | Frontend styling and responsive user interface design.                                                              |
| `↳ script.js`                               | Frontend interaction logic and communication with the Flask backend.                                                |
| `↳ requirements.txt`                        | Python dependencies required for running the web application locally.                                               |
| `TESS_dataset/`                             | Contains both compressed and extracted versions of the TESS emotional speech dataset.                               |
| `↳ archive.zip`                             | Original compressed TESS dataset archive used during preprocessing.                                                 |
| `↳ TESS Toronto emotional speech set data/` | Extracted dataset directory containing all emotional speech samples.                                                |
| `MELD_text_model/`                  | Standalone RoBERTa-based text emotion recognition pipeline trained and evaluated on the MELD conversational emotion dataset. |
| `↳ MELD_dataset/`                  | Contains MELD train, validation, and test CSV files used for conversational text emotion recognition. |
| `↳ train.py`                       | Trains the standalone MELD text emotion recognition model using RoBERTa embeddings. |
| `↳ test.py`                        | Evaluates the trained MELD model and generates classification reports, confusion matrices, and t-SNE visualizations. |
| `↳ requirements.txt`               | Python dependencies required for running the standalone MELD text model pipeline. |
| `↳ best_meld_text_model.pth`       | Trained MELD text emotion recognition model checkpoint. |
| `↳ learning_curve.png`             | Training loss and validation accuracy visualization generated during MELD model training. |
| `↳ confusion_matrix.png`           | Confusion matrix visualization for MELD text emotion predictions. |
| `↳ tsne_plot.png`                  | t-SNE latent embedding visualization for MELD text representations. |
| `↳ classification_report.txt`      | Precision, recall, F1-score, and accuracy metrics for the MELD text model. |
| `best_speech_model.pth`                     | Pretrained Speech-Only model checkpoint.                                                                            |
| `best_text_model.pth`                       | Pretrained Text-Only model checkpoint.                                                                              |
| `best_fusion_model.pth`                     | Pretrained Multimodal Fusion model checkpoint.                                                                      |
| `.gitignore`                                | Prevents large datasets, model checkpoints, cache files, and system files from being tracked by Git.                |
| `README.md`                                 | Project documentation, architecture explanations, setup instructions, experimental results, and workflow details.   |
| `Report.pdf`                                | Complete technical report containing architecture design, experiments, analysis, observations, and conclusions.     |

---

<a id="clone-repo"></a>

### ⬇️ 4. Clone Repository

Clone the project repository from GitHub:

```bash
git clone https://github.com/vishakhmv/IIIT_project.git
```

Move into the project directory:

```bash
cd IIIT_project
```

---

<a id="after-cloning"></a>

#### 📂 5. After Cloning

After cloning the repository, place the dataset and model weights in the correct directory structure

The final directory structure should match the structure shown in the **Project Structure** section.

---

#### ⚠️ Important Note

The GitHub repository does **not** include:

- pretrained `.pth` model checkpoints
- the TESS dataset
- extracted audio files

These large files are intentionally excluded using `.gitignore`.

Please download them separately using the links provided in the **Download Requirements** section before running the project.

---

<a id="install-dependencies"></a>

### 🛠️ 6. Install Dependencies

Install all required Python libraries using:

```bash
pip install -r project/requirements.txt
```

---

#### 📦 Dependencies

The project primarily uses the following libraries and frameworks:

| Library        | Purpose                                                  |
| -------------- | -------------------------------------------------------- |
| `PyTorch`      | Deep learning framework for model training and inference |
| `Transformers` | Provides pretrained HuBERT and DistilBERT models         |
| `Librosa`      | Audio loading and MFCC feature extraction                |
| `Scikit-learn` | Dataset splitting, evaluation metrics, and t-SNE         |
| `Pandas`       | CSV handling and structured data processing              |
| `NumPy`        | Numerical computations and tensor preparation            |
| `Matplotlib`   | Plot generation and visualization                        |
| `Seaborn`      | Confusion matrix and statistical visualizations          |

---

#### ⚠️ Important Note

The first execution may automatically download pretrained Hugging Face models including:

- `facebook/hubert-base-ls960`
- `distilbert-base-uncased`

Ensure that your system has an active internet connection during the initial setup.

---

<a id="training"></a>

### 🚀 7. Run Training

> **Optional:**  
> This step is only required if you want to retrain the models from scratch.
>
> If you only want to evaluate the pretrained models, you can directly proceed to the **🧪 Run Testing** section.

---

#### 🎙️ Train Speech-Only Model

```bash
python project/models/speech_pipeline/train.py
```

---

#### 📝 Train Text-Only Model

```bash
python project/models/text_pipeline/train.py
```

---

#### 🔀 Train Multimodal Fusion Model

```bash
python project/models/fusion_pipeline/train.py
```

---

#### 📌 Training Outputs

Training automatically generates:

- pretrained model checkpoints (`.pth`)
- learning curve visualizations

Model checkpoints are stored in:

```bash
IIIT_project/
├── best_speech_model.pth
├── best_text_model.pth
└── best_fusion_model.pth
```

Learning curve plots are stored in:

```bash
IIIT_project/
└── project/
    └── Results/
        └── plots/
            ├── Speech_model/
            ├── Text_model/
            └── Fusion_model/
```

---

<a id="testing"></a>

### 🧪 8. Run Testing

The testing pipelines evaluate pretrained models on unseen test samples and automatically generate:

- classification accuracy tables
- confusion matrices
- t-SNE latent space visualizations

---

#### 🎙️ Test Speech-Only Model

```bash
python -B project/models/speech_pipeline/test.py
```

---

#### 📝 Test Text-Only Model

```bash
python -B project/models/text_pipeline/test.py
```

---

#### 🔀 Test Multimodal Fusion Model

```bash
python -B project/models/fusion_pipeline/test.py
```

---

> **💡 Note on the `-B` flag:** The the `python -B` command prevents Python from generating `__pycache__` folders, keeping the project directory clean during evaluation.

---

#### 📌 Testing Outputs

Testing automatically generates:

- classification reports (`.csv`)
- confusion matrices
- t-SNE visualizations

---

##### 📊 Classification Reports

Generated classification reports are stored in:

```bash
IIIT_project/
└── project/
    └── Results/
        ├── speech_classification_report.csv
        ├── text_classification_report.csv
        └── fusion_classification_report.csv
```

---

##### 🔥 Confusion Matrices & t-SNE Visualizations

Generated plots are stored in:

```bash
IIIT_project/
└── project/
    └── Results/
        └── plots/
            ├── Speech_model/
            │   ├── confusion_matrix.png
            │   └── tsne.png
            │
            ├── Text_model/
            │   ├── confusion_matrix.png
            │   └── tsne.png
            │
            └── Fusion_model/
                ├── confusion_matrix.png
                └── tsne.png
```

---

#### 📈 Generated Evaluation Artifacts

Each testing pipeline produces:

| Artifact                    | Description                                        |
| --------------------------- | -------------------------------------------------- |
| `classification_report.csv` | Class-wise Precision, Recall, and F1-Score metrics |
| `confusion_matrix.png`      | Class-wise prediction performance visualization    |
| `tsne.png`                  | Latent feature-space clustering visualization      |

---

#### 🎥 Demo Evaluation Videos

Example testing demonstrations for all three architectures:

| Model                      | Demo Video                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| 🎙️ Speech-Only Model       | [Watch Demo](https://drive.google.com/file/d/1JdgQiZig_IydqLO_BgVXBRuZDKguvfAR/view?usp=drive_link) |
| 📝 Text-Only Model         | [Watch Demo](https://drive.google.com/file/d/1N0V3SBaDMd1nXcsnOz-80AXdc2nAGugd/view?usp=drive_link) |
| 🔀 Multimodal Fusion Model | [Watch Demo](https://drive.google.com/file/d/1DCMvcD3ghM_wx8UfSQTg3Hy2_BjBy71J/view?usp=drive_link) |

---

---

<a id="meld-text-model"></a>

### 💬 9. Standalone MELD Text Emotion Recognition Model

In addition to the primary TESS-based Speech, Text, and Multimodal Fusion architectures, this repository also includes a standalone conversational text emotion recognition pipeline trained on the MELD dataset using a RoBERTa-based transformer architecture.

The MELD module is fully self-contained inside:

```bash
IIIT_project/
└── MELD_text_model/
```

---

#### 📦 9.1 MELD Dataset Setup

Download the official MELD dataset from:

- https://affective-meld.github.io/

After downloading, place the dataset CSV files inside:

```bash
IIIT_project/
└── MELD_text_model/
    └── MELD_dataset/
```

---
Before continuing, make sure you are inside the root project directory in the terminal:

#### 🚀 9.2 Train MELD Text Model *(Optional)*

Run the training pipeline using:

```bash
python MELD_text_model/train.py
```

Training automatically generates:

- pretrained model checkpoint
- learning curve visualization

Generated outputs:

```bash
IIIT_project/
└── MELD_text_model/
    ├── best_meld_text_model.pth
    └── learning_curve.png
```

---

#### 🧪 9.3 Test MELD Text Model

##### 🧠 Pretrained MELD Model Checkpoint

Download the pretrained MELD text model checkpoint from:

- https://drive.google.com/file/d/1LsvYrzwkiDa_hhg0Xia1VC1zQBUBVlqU/view?usp=drive_link

Place the checkpoint inside:

```bash
IIIT_project/
└── MELD_text_model/
    └── best_meld_text_model.pth
```

---

##### 🛠️ Install MELD Dependencies

Install all required Python libraries using:

```bash
pip install -r MELD_text_model/requirements.txt
```

---

##### 🧪 Rinning test.py

Run the evaluation pipeline using:

```bash
python -B MELD_text_model/test.py
```

Testing automatically generates:

- classification report
- confusion matrix
- t-SNE embedding visualization

Generated outputs:

```bash
IIIT_project/
└── MELD_text_model/
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── tsne_plot.png
```

---

#### 📊 9.4 Generated Evaluation Artifacts

| Artifact                    | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `classification_report.txt` | Precision, Recall, F1-Score, and Accuracy metrics     |
| `confusion_matrix.png`      | Emotion prediction confusion matrix visualization     |
| `tsne_plot.png`             | Latent embedding clustering visualization using t-SNE |
| `learning_curve.png`        | Training loss and validation accuracy visualization   |

---

<a id="web-app"></a>

### 🌐 10. Run Web Application

The project includes a fully functional and responsive web application named **Bhavora AI**, built using a Python (Flask) backend with an HTML/CSS/JavaScript frontend.

#### ✨ The Meaning of "Bhavora"

The name **Bhavora** is inspired by the Sanskrit word **_“Bhava”_** (भाव), meaning emotion or feeling, combined with the word **_“Aura”_** representing an emotional presence or atmosphere. Together, Bhavora symbolizes an “Emotional Aura,” reflecting the system’s ability to recognize human emotions from subtle patterns in speech and text.

The web interface allows real-time emotion prediction using the pretrained Speech-Only, Text-Only, and Multimodal Fusion architectures through an interactive dashboard.

---

#### 🛠️ Install Web Dependencies

The web application uses a separate lightweight dependency configuration to keep the inference environment isolated from the training environment.

Move into the `Web/` directory:

```bash
cd Web
```

Install all required web application dependencies:

```bash
pip install -r requirements.txt
```

---

#### 📦 Web Dependencies

To maintain a lightweight inference environment independent of the training pipeline, the web application uses a separate `requirements.txt` file. The backend server relies on the following packages:

| Library        | Version Requirement | Purpose                                                                                    |
| :------------- | :------------------ | :----------------------------------------------------------------------------------------- |
| `torch`        | `>=2.0.0`           | Deep learning framework used for loading pretrained model weights and performing inference |
| `transformers` | `>=4.35.0`          | Provides pretrained transformer architectures including DistilBERT and HuBERT              |
| `flask`        | `>=3.0.0`           | Lightweight backend web framework used to serve the inference application                  |
| `librosa`      | `>=0.10.0`          | Audio processing library used for waveform loading and MFCC feature extraction             |
| `soundfile`    | `>=0.12.1`          | Handles decoding and processing of uploaded audio files                                    |
| `numpy`        | `>=1.24.0`          | Numerical computation library used for tensor preparation and preprocessing                |
| `Flask-CORS`   | `>=4.0.0`           | Enables secure communication between the frontend interface and backend server             |

---

#### 🚀 Start Flask Backend

Launch the local Flask development server:

```bash
python app.py
```

The backend server will automatically initialize on your local machine.

---

#### 🌍 Open the Web Interface

Once the Flask server starts successfully, open your web browser and navigate to:

```text
http://127.0.0.1:5000
```

---

#### 🎭 Supported Prediction Modes

The web dashboard supports real-time emotion prediction for all three architectures:

| Mode                 | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| 🎙️ Speech-Only       | Upload `.wav` or `.flac` audio files for speech emotion recognition    |
| 📝 Text-Only         | Enter text input for semantic emotion prediction                       |
| 🔀 Multimodal Fusion | Provide both speech and text inputs for multimodal emotion recognition |

---

#### 🖥️ Web Interface Preview

The web application includes:

- Responsive user interface
- Real-time prediction dashboard
- Dark/Light theme toggle
- Probability visualization
- Multimodal prediction support

<p align="center">
  <img src="assets/web-light.png" width="45%">
  <img src="assets/web-dark.png" width="45%">
</p>

---

#### 🎥 Web Application Demonstration

Watch the real-time emotion recognition system in action:

- [Watch Web Application Demo](https://drive.google.com/file/d/1S3PBq-C0EeLMSWoC8bE-xJpc2mXzv23d/view?usp=drive_link)

---
