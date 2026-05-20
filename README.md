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

The generated metrics include:

- Precision
- Recall
- F1-Score
- Overall Accuracy
- Macro Average
- Weighted Average

Generated output:

```bash
IIIT_project/
└── Results/
    └── speech_accuracy_table.csv
```

<p align="center">
  <img src="assets/speech-accuracy-table.png" width="75%">
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

### 📝 Text-Only Model

The Text-Only pipeline predicts emotions using text.

It learns semantic and contextual language representations directly from textual input using a pretrained transformer-based language model.

---

#### 🏗️ a. System Architecture

The Text-Only architecture performs emotion classification using pretrained transformer-based contextual language embeddings generated by DistilBERT.

<p align="center">
  <img src="assets/text-model-architecture.png" width="90%">
</p>

---

##### 📝 Input Text

The model receives textual input sentences representing reconstructed speech transcripts.

Example:

```text
"say the word back"
```

These text sequences serve as the only input modality for the Text-Only architecture.

---

##### 🔤 Text Tokenization

The input text is tokenized using the pretrained:

```python
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```

tokenizer.

The tokenizer converts raw text into transformer-compatible representations including:

- Input Token IDs
- Attention Masks
- Special Tokens (`[CLS]`, `[SEP]`)

All sequences are padded and truncated to a fixed sequence length for consistent batch processing.

---

##### 🧠 DistilBERT Contextual Embeddings

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

##### 🎯 CLS Token Representation

After transformer encoding, the architecture extracts the contextual embedding corresponding to the `[CLS]` token:

```python
outputs.last_hidden_state[:, 0, :]
```

The CLS embedding acts as a compact sentence-level representation summarizing the semantic meaning of the entire input sequence.

---

##### 🔗 Classification Head

The CLS embedding is passed through a fully connected classification head consisting of:

- Linear Layers
- Layer Normalization
- ReLU Activation
- Dropout Regularization

The classification head maps semantic language embeddings into emotional category probabilities.

---

##### 🎭 Emotion Classification

The architecture predicts one of the following seven emotional categories:

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

The final output of the Text-Only architecture is a probability distribution across all emotional categories.

The emotion with the highest probability score is selected as the predicted emotional state of the input text sequence.


#### 🏋️ b. Training Pipeline

The Text-Only training pipeline learns emotional representations directly from textual input using pretrained transformer-based contextual embeddings generated by DistilBERT.

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

The generated metrics include:

- Precision
- Recall
- F1-Score
- Overall Accuracy
- Macro Average
- Weighted Average

Generated output:

```bash
IIIT_project/
└── Results/
    └── text_accuracy_table.csv
```

<p align="center">
  <img src="assets/text-accuracy-table.png" width="75%">
</p>

The Text-Only model achieved an overall test accuracy of approximately **14.28%**, with extremely low precision, recall, and F1-scores across most emotional categories.

This performance is close to random guessing across seven classes, indicating that semantic text information alone is insufficient for reliable emotion recognition in the TESS dataset.

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

The TESS dataset uses nearly identical carrier phrases across all emotional categories.

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
