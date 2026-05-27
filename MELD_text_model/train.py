import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

try:

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

print(f"🚀 RUNNING FROM PORTABLE DIRECTORY: {CURRENT_DIR}")

MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "best_meld_text_model.pth")
PLOT_SAVE_PATH = os.path.join(CURRENT_DIR, "learning_curve.png")

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LEN = 128

EMOTION_MAP = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

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

def load_data():
    print("Searching for MELD dataset...")
    train_file = next((os.path.join(root, name) for root, dirs, files in os.walk(CURRENT_DIR) for name in files if 'train' in name.lower() and name.endswith('.csv')), None)
    dev_file = next((os.path.join(root, name) for root, dirs, files in os.walk(CURRENT_DIR) for name in files if 'dev' in name.lower() and name.endswith('.csv')), None)

    if not train_file or not dev_file:
        raise FileNotFoundError("Could not find train/dev CSV files. Ensure MELD_dataset folder is here.")

    print(f"Found Train Data: {train_file}")
    print(f"Found Dev Data: {dev_file}")

    train_df = pd.read_csv(train_file).dropna(subset=['Utterance', 'Emotion'])
    val_df = pd.read_csv(dev_file).dropna(subset=['Utterance', 'Emotion'])

    train_df['label'] = train_df['Emotion'].str.lower().map(EMOTION_MAP)
    val_df['label'] = val_df['Emotion'].str.lower().map(EMOTION_MAP)

    return train_df, val_df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")
    
    train_df, val_df = load_data()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_dataset = MELDDataset(train_df['Utterance'].values, train_df['label'].values, tokenizer, MAX_LEN)
    val_dataset = MELDDataset(val_df['Utterance'].values, val_df['label'].values, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(EMOTION_MAP))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_accuracies = [], []
    best_acc = 0.0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   --> Saved new best model to {MODEL_SAVE_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, EPOCHS+1), val_accuracies, label='Validation Accuracy', color='orange', marker='s')
    plt.title('Training Loss and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    print(f"Learning curve permanently saved to Drive at: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()
