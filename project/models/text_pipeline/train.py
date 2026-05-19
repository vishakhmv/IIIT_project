import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel

# Emotion label mapping
EMOTION_MAP = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'pleasant': 5, 'sad': 6}

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=256):
        super(TextEmotionModel, self).__init__()
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT parameters so we only train the classification head
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        # Extract CLS token embedding (sentence representation)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

class TESSTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=16):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        label = int(self.df.iloc[idx]['label_encoded'])
        
        # Use the modern Hugging Face __call__ method instead of encode_plus
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Text-Only training on device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths for data and saving models
    train_csv_path = os.path.normpath(os.path.join(script_dir, '../../../Data_split/train_split.csv'))
    pth_save_path = os.path.normpath(os.path.join(script_dir, '../../../best_text_model.pth'))
    plot_save_path = os.path.normpath(os.path.join(script_dir, '../../Results/plots/Text_model/learning_curves.png'))
    
    # Load training data
    df_train_all = pd.read_csv(train_csv_path)
    
    # Split into train and validation sets
    df_train, df_val = train_test_split(
        df_train_all, test_size=0.15, stratify=df_train_all['label_encoded'], random_state=42
    )
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_loader = DataLoader(TESSTextDataset(df_train, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(TESSTextDataset(df_val, tokenizer), batch_size=32, shuffle=False)
    
    model = TextEmotionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=1e-2)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    epochs = 15
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * input_ids.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * input_ids.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                
        t_loss = train_loss / len(train_loader.dataset)
        t_acc = train_correct / len(train_loader.dataset)
        v_loss = val_loss / len(val_loader.dataset)
        v_acc = val_correct / len(val_loader.dataset)
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} | Val Acc: {v_acc*100:.2f}%")
        
        # Save the best model weights
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), pth_save_path)
            
    # Generate and save learning curves
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title('Text Model: Loss Profiles')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy', marker='s')
    ax2.set_title('Text Model: Accuracy Profiles')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Learning curves chart saved directly to: {plot_save_path}")
    print(f"[SUCCESS] Best Text weights saved directly to: {pth_save_path}")

if __name__ == '__main__':
    train_pipeline()
