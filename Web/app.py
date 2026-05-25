import os
import sys
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import librosa
import tempfile
import traceback
from flask import Flask, request, jsonify, send_from_directory
from transformers import HubertModel, DistilBertModel, DistilBertTokenizer


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR   = os.path.dirname(BASE_DIR) 

SPEECH_PTH = os.path.join(ROOT_DIR, 'best_speech_model.pth')
TEXT_PTH   = os.path.join(ROOT_DIR, 'best_text_model.pth')
FUSION_PTH = os.path.join(ROOT_DIR, 'best_fusion_model.pth')
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant surprise', 'sad']
SAMPLE_RATE    = 16000
MAX_SAMPLES    = int(3.0 * SAMPLE_RATE)
MAX_TEXT_LEN   = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[BHAVORA] Device: {device}')

for name, path in [('Speech', SPEECH_PTH), ('Text', TEXT_PTH), ('Fusion', FUSION_PTH)]:
    if not os.path.exists(path):
        print(f'[ERROR] {name} model not found at: {path}')
        sys.exit(1)

class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes=7, mfcc_features=20, hidden_dim=256, num_layers=2):
        super().__init__()
        self.hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        for p in self.hubert.parameters():
            p.requires_grad = False
        self.speech_temporal = nn.LSTM(
            input_size=768 + mfcc_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_raw, x_mfcc):
        with torch.no_grad():
            h = self.hubert(x_raw).last_hidden_state
        t = min(h.size(1), x_mfcc.size(1))
        out, _ = self.speech_temporal(torch.cat((h[:, :t], x_mfcc[:, :t]), dim=-1))
        return self.classifier(torch.mean(out, dim=1))


class TextEmotionModel(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=7, mfcc_features=20, hidden_dim=256, num_layers=2):
        super().__init__()
        self.hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        for p in self.hubert.parameters():
            p.requires_grad = False
        self.speech_temporal = nn.LSTM(
            input_size=768 + mfcc_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * 2) + self.bert.config.hidden_size, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x_raw, x_mfcc, input_ids, attention_mask):
        with torch.no_grad():
            h = self.hubert(x_raw).last_hidden_state
        t = min(h.size(1), x_mfcc.size(1))
        lstm_out, _ = self.speech_temporal(torch.cat((h[:, :t], x_mfcc[:, :t]), dim=-1))
        speech_feat = torch.mean(lstm_out, dim=1)
        with torch.no_grad():
            text_feat = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat((speech_feat, text_feat), dim=1))


print('[BHAVORA] Loading Speech model...')
speech_model = SpeechEmotionModel().to(device)
speech_model.load_state_dict(torch.load(SPEECH_PTH, map_location=device))
speech_model.eval()
print('[BHAVORA]   Speech model loaded.')

print('[BHAVORA] Loading Text model...')
text_model = TextEmotionModel().to(device)
text_model.load_state_dict(torch.load(TEXT_PTH, map_location=device))
text_model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print('[BHAVORA]   Text model loaded.')

print('[BHAVORA] Loading Fusion model...')
fusion_model = MultimodalFusionModel().to(device)
fusion_model.load_state_dict(torch.load(FUSION_PTH, map_location=device))
fusion_model.eval()
print('[BHAVORA]   Fusion model loaded.')

def preprocess_audio(file_storage):
    suffix = os.path.splitext(file_storage.filename)[-1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE)
    finally:
        os.remove(tmp_path)

    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode='constant')
    else:
        y = y[:MAX_SAMPLES]

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20, hop_length=320)
    raw  = torch.FloatTensor(y).unsqueeze(0).to(device)
    mfcc = torch.FloatTensor(mfcc.T).unsqueeze(0).to(device)
    return raw, mfcc


def preprocess_text(text):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_TEXT_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def decode(logits):
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    all_probs = {EMOTION_LABELS[i]: round(probs[i].item() * 100, 2) for i in range(len(EMOTION_LABELS))}
    return EMOTION_LABELS[idx.item()], conf.item(), all_probs


app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route('/predict/speech', methods=['POST'])
def predict_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    try:
        raw, mfcc = preprocess_audio(request.files['audio'])
        with torch.no_grad():
            logits = speech_model(raw, mfcc)
        emotion, confidence, all_probs = decode(logits)
        print(f'[SPEECH] {emotion} ({confidence*100:.2f}%)')
        return jsonify({'emotion': emotion, 'confidence': confidence, 'all_probs': all_probs})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': traceback.format_exc()}), 500


@app.route('/predict/text', methods=['POST'])
def predict_text():
    data = request.get_json(silent=True)
    if not data or not data.get('text', '').strip():
        return jsonify({'error': 'No text provided'}), 400
    try:
        ids, mask = preprocess_text(data['text'])
        with torch.no_grad():
            logits = text_model(ids, mask)
        emotion, confidence, all_probs = decode(logits)
        print(f'[TEXT] {emotion} ({confidence*100:.2f}%)')
        return jsonify({'emotion': emotion, 'confidence': confidence, 'all_probs': all_probs})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': traceback.format_exc()}), 500


@app.route('/predict/fusion', methods=['POST'])
def predict_fusion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    text = request.form.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        raw, mfcc = preprocess_audio(request.files['audio'])
        ids, mask = preprocess_text(text)
        with torch.no_grad():
            logits = fusion_model(raw, mfcc, ids, mask)
        emotion, confidence, all_probs = decode(logits)
        print(f'[FUSION] {emotion} ({confidence*100:.2f}%)')
        return jsonify({'emotion': emotion, 'confidence': confidence, 'all_probs': all_probs})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
