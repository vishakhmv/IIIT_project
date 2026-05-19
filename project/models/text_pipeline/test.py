import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer
# Import the architecture and dataset loader directly from our train script
from train import TextEmotionModel, TESSTextDataset

def test_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Text-Only evaluation on device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define relative paths for loading and saving
    test_csv_path = os.path.normpath(os.path.join(script_dir, '../../../Data_split/test_split.csv'))
    pth_load_path = os.path.normpath(os.path.join(script_dir, '../../../best_text_model.pth'))
    
    csv_out_path = os.path.normpath(os.path.join(script_dir, '../../Results/text_accuracy_table.csv'))
    matrix_out_path = os.path.normpath(os.path.join(script_dir, '../../Results/plots/Text_model/confusion_matrix.png'))
    
    # Load test data and tokenizer
    df_test = pd.read_csv(test_csv_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_loader = DataLoader(TESSTextDataset(df_test, tokenizer), batch_size=32, shuffle=False)
    
    # Load the trained model weights
    model = TextEmotionModel()
    if not os.path.exists(pth_load_path):
        print(f"Error: Could not find weights at {pth_load_path}")
        return
        
    model.load_state_dict(torch.load(pth_load_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Evaluating test set... (Extracting text features and running inference)")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Fixed alphabetical dictionary mapping
    target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant surprise', 'sad']
    
    # Calculate evaluation metrics and save as .csv
    report_dict = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    df_metrics = pd.DataFrame(report_dict).transpose()

    df_metrics.index.name = 'emotion'
    
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    df_metrics.to_csv(csv_out_path, index=True)
    print(f"\n[SUCCESS] Text variance accuracy table successfully saved to: {csv_out_path}")
    
    # Generate Seaborn heatmap confusion matrix in standard Blue
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs(os.path.dirname(matrix_out_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Text-Only Model Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    
    plt.tight_layout()
    plt.savefig(matrix_out_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Text confusion matrix plot successfully saved to: {matrix_out_path}")

if __name__ == '__main__':
    test_pipeline()
