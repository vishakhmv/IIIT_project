import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

print(f"RUNNING FROM PORTABLE DIRECTORY: {CURRENT_DIR}")

MODEL_PATH = os.path.join(CURRENT_DIR, "best_meld_text_model.pth")
CM_SAVE_PATH = os.path.join(CURRENT_DIR, "confusion_matrix.png")
TSNE_SAVE_PATH = os.path.join(CURRENT_DIR, "tsne_plot.png")
REPORT_SAVE_PATH = os.path.join(CURRENT_DIR, "classification_report.txt")

BATCH_SIZE = 16
MAX_LEN = 128

EMOTION_MAP = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
REVERSE_MAP = {v: k for k, v in EMOTION_MAP.items()}
CLASS_NAMES = [REVERSE_MAP[i] for i in range(len(EMOTION_MAP))]

class MELDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_test_data():
    print("Locating test dataset...")
    test_file = next((os.path.join(root, name) for root, dirs, files in os.walk(CURRENT_DIR) for name in files if 'test' in name.lower() and name.endswith('.csv')), None)
    
    if not test_file:
        raise FileNotFoundError("Could not find test CSV file. Ensure MELD_dataset folder is here.")
    
    print(f"Found Test Data: {test_file}")
    test_df = pd.read_csv(test_file).dropna(subset=['Utterance', 'Emotion'])
    test_df['label'] = test_df['Emotion'].str.lower().map(EMOTION_MAP)
    return test_df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    test_df = load_test_data()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_dataset = MELDDataset(test_df['Utterance'].values, test_df['label'].values, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(EMOTION_MAP), output_hidden_states=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels, all_embeddings = [], [], []

    print("Evaluating model and extracting embeddings... (This takes a moment)")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            hidden_states = outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.append(cls_embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)

    present_labels = sorted(list(set(all_labels) | set(all_preds)))
    present_class_names = [CLASS_NAMES[i] for i in present_labels]

    report = classification_report(all_labels, all_preds, labels=present_labels, target_names=present_class_names, zero_division=0)
    print("\nClassification Report:\n", report)
    with open(REPORT_SAVE_PATH, "w") as f:
        f.write(report)
    print(f"Saved classification report to {REPORT_SAVE_PATH}")

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASS_NAMES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(CM_SAVE_PATH)
    plt.close()
    print(f"Saved confusion matrix to {CM_SAVE_PATH}")

    print("Computing t-SNE... (this may take a minute)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=[CLASS_NAMES[label] for label in all_labels],
        palette=sns.color_palette("hsv", len(CLASS_NAMES)),
        legend="full", alpha=0.7
    )
    plt.title('t-SNE Projection of Text Embeddings')
    plt.savefig(TSNE_SAVE_PATH)
    plt.close()
    print(f"Saved t-SNE plot to {TSNE_SAVE_PATH}")

if __name__ == "__main__":
    main()
