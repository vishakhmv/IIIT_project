import os
import torch
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
# Import the architecture and dataset loader directly from our train script
from train import SpeechEmotionModel, TESSSpeechDataset


def save_tsne_plot(features, labels, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    # 'hue' uses the text labels to create a beautiful, named legend
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="deep")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def test_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Speech-Only evaluation on device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define relative paths for loading and saving
    test_csv_path = os.path.normpath(os.path.join(script_dir, '../../../Data_split/test_split.csv'))
    base_audio_dir = os.path.normpath(os.path.join(script_dir, '../../../TESS_dataset/TESS Toronto emotional speech set data'))
    pth_load_path = os.path.normpath(os.path.join(script_dir, '../../../best_speech_model.pth'))
    
    csv_out_path = os.path.normpath(os.path.join(script_dir, '../../Results/speech_accuracy_table.csv'))
    matrix_out_path = os.path.normpath(os.path.join(script_dir, '../../Results/plots/Speech_model/confusion_matrix.png'))
    tsne_out_path = os.path.normpath(os.path.join(script_dir, '../../Results/plots/Speech_model/tsne.png'))
    
    # Pre-flight structural sanity check for audio dir
    if not os.path.exists(base_audio_dir):
        alt_dir = os.path.normpath(os.path.join(script_dir, '../../../TESS_dataset/tess toronto emotional speech set data'))
        if os.path.exists(alt_dir):
            base_audio_dir = alt_dir
    
    # Load test data
    df_test = pd.read_csv(test_csv_path)
    test_loader = DataLoader(TESSSpeechDataset(df_test, base_audio_dir), batch_size=32, shuffle=False)
    
    # Load the trained model weights
    model = SpeechEmotionModel()
    if not os.path.exists(pth_load_path):
        print(f"Error: Could not find weights at {pth_load_path}")
        return
        
    model.load_state_dict(torch.load(pth_load_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_features = []
    
    print("Evaluating test set... (Extracting acoustic features and running inference)")
    with torch.no_grad():
        for batch in test_loader:
            raw_audio = batch['raw_audio'].to(device)
            mfcc = batch['mfcc'].to(device)
            targets = batch['targets'].to(device)
    
            hubert_out = model.hubert(raw_audio).last_hidden_state
            target_time = min(hubert_out.size(1), mfcc.size(1))
            combined = torch.cat((hubert_out[:, :target_time, :], mfcc[:, :target_time, :]), dim=-1)
            lstm_out, _ = model.speech_temporal(combined)
            features = torch.mean(lstm_out, dim=1) 
            outputs = model.classifier(features)
            all_features.append(features.cpu())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Fixed alphabetical dictionary mapping
    target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant surprise', 'sad']

    print("Generating t-SNE visualization...")
    
    all_features_numpy = torch.cat(all_features, dim=0).numpy()
    
    named_labels = [target_names[label] for label in all_targets]
    
    save_tsne_plot(all_features_numpy, named_labels, "t-SNE Latent Space: Speech BiLSTM", tsne_out_path)
    print(f"[SUCCESS] Speech t-SNE plot successfully saved to: {tsne_out_path}")
    
    # Calculate evaluation metrics and save as .csv
    report_dict = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    df_metrics = pd.DataFrame(report_dict).transpose()

    df_metrics.index.name = 'emotion'
    
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    df_metrics.to_csv(csv_out_path, index=True)
    print(f"\n[SUCCESS] Speech variance accuracy table successfully saved to: {csv_out_path}")
    
    # Generate Seaborn heatmap confusion matrix in standard Blue
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs(os.path.dirname(matrix_out_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Speech-Only Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    
    plt.tight_layout()
    plt.savefig(matrix_out_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Speech confusion matrix plot successfully saved to: {matrix_out_path}")

if __name__ == '__main__':
    test_pipeline()
