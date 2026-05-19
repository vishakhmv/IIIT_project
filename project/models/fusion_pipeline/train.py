import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel, HubertModel

# Emotion label mapping
EMOTION_MAP = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'pleasant': 5, 'sad': 6}

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=7, mfcc_features=20, hidden_dim=256, num_layers=2):
        super(MultimodalFusionModel, self).__init__()
        
        # --- SPEECH BRANCH ---
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        for param in self.hubert.parameters():
            param.requires_grad = False
            
        self.speech_temporal = nn.LSTM(
            input_size=768 + mfcc_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        # --- TEXT BRANCH ---
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # --- FUSION & CLASSIFIER BRANCH ---
        # Audio BiLSTM outputs 512 (256 * 2), Text DistilBERT outputs 768
        fusion_dim = (hidden_dim * 2) + self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x_raw, x_mfcc, input_ids, attention_mask):
        # 1. Process Speech
        with torch.no_grad():
            hubert_outputs = self.hubert(x_raw).last_hidden_state
        target_time_steps = min(hubert_outputs.size(1), x_mfcc.size(1))
        
        hubert_feats = hubert_outputs[:, :target_time_steps, :]
        mfcc_feats = x_mfcc[:, :target_time_steps, :]
        combined_audio = torch.cat((hubert_feats, mfcc_feats), dim=-1)
        
        lstm_out, _ = self.speech_temporal(combined_audio)
        pooled_speech = torch.mean(lstm_out, dim=1)
        
        # 2. Process Text
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text = bert_outputs.last_hidden_state[:, 0, :]
        
        # 3. Multimodal Fusion (Concatenation)
        fused_features = torch.cat((pooled_speech, pooled_text), dim=1)
        
        # 4. Classification
        return self.classifier(fused_features)

class TESSMultimodalDataset(Dataset):
    def __init__(self, dataframe, base_audio_dir, tokenizer, max_duration=3.0, sample_rate=16000, max_length=16):
        self.df = dataframe
        self.base_audio_dir = base_audio_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label_encoded'])
        
        # Audio Extraction
        audio_path = os.path.join(self.base_audio_dir, row['path'])
        y, _ = librosa.load(audio_path, sr=self.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=25)
        
        if len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)), mode='constant')
        else:
            y = y[:self.max_samples]
            
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=20, hop_length=320)
        
        # Text Extraction
        text = str(row['text'])
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
            'raw_audio': torch.FloatTensor(y),
            'mfcc': torch.FloatTensor(mfcc.T),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Multimodal Fusion training on device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths for loading and saving
    train_csv_path = os.path.normpath(os.path.join(script_dir, '../../../Data_split/train_split.csv'))
    base_audio_dir = os.path.normpath(os.path.join(script_dir, '../../../TESS_dataset/TESS Toronto emotional speech set data'))
    pth_save_path = os.path.normpath(os.path.join(script_dir, '../../../best_fusion_model.pth'))
    plot_save_path = os.path.normpath(os.path.join(script_dir, '../../Results/plots/Fusion_model/learning_curves.png'))
    
    # Load and split dataset
    df_train_all = pd.read_csv(train_csv_path)
    df_train, df_val = train_test_split(df_train_all, test_size=0.15, stratify=df_train_all['label_encoded'], random_state=42)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_loader = DataLoader(TESSMultimodalDataset(df_train, base_audio_dir, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(TESSMultimodalDataset(df_val, base_audio_dir, tokenizer), batch_size=32, shuffle=False)
    
    model = MultimodalFusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for batch in train_loader:
            raw_audio = batch['raw_audio'].to(device)
            mfcc = batch['mfcc'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(raw_audio, mfcc, input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * raw_audio.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                raw_audio = batch['raw_audio'].to(device)
                mfcc = batch['mfcc'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(raw_audio, mfcc, input_ids, attention_mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * raw_audio.size(0)
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
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), pth_save_path)
            
    # Generate and save learning curves
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title('Fusion Model: Loss Profiles')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(range(1, epochs + 1), history['val_acc'], label='Val Acc', marker='s')
    ax2.set_title('Fusion Model: Accuracy Profiles')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Fusion charts saved directly to: {plot_save_path}")
    print(f"[SUCCESS] Best Fusion weights saved directly to: {pth_save_path}")

if __name__ == '__main__':
    train_pipeline()
