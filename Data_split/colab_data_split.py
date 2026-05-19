import os
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTION_MAP = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'pleasant': 5, 'sad': 6}

# Base dataset path where audio was extracted locally
base_audio_dir = '/content/tess_dataset/TESS Toronto emotional speech set data'

data = []

# Crawl through the local dataset and build the strict data structures
for root, dirs, files in os.walk(base_audio_dir):
    for file in files:
        if file.endswith('.wav'):
            rel_path = os.path.relpath(os.path.join(root, file), base_audio_dir)

            # Extract emotion from folder name safely
            folder_name = os.path.basename(root)
            if '_' in folder_name:
                raw_emotion = folder_name.split('_')[1].lower()
                # Handle the 'pleasant_surprise' edge case mapping it to 'pleasant'
                emotion = 'pleasant' if 'pleasant' in raw_emotion else raw_emotion

                if emotion in EMOTION_MAP:
                    # Extract the target word from the filename (e.g., 'YAF_back_angry.wav' -> 'back')
                    name_parts = os.path.splitext(file)[0].split('_')
                    word = name_parts[1].lower() if len(name_parts) >= 3 else "unknown"
                    text_phrase = f"say the word {word}"

                    data.append({
                        'path': rel_path,
                        'text': text_phrase,
                        'label_encoded': EMOTION_MAP[emotion]
                    })

# Create the DataFrame
df = pd.DataFrame(data)

# Perform a Stratified Split (80% Training, 20% Testing) to ensure balanced classes in both sets
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label_encoded'], random_state=42
)

# Save the splits to the required Data_split directory in Google Drive
data_split_dir = '/content/drive/MyDrive/IIIT_project/Data_split'
os.makedirs(data_split_dir, exist_ok=True)

train_csv_path = os.path.join(data_split_dir, 'train_split.csv')
test_csv_path = os.path.join(data_split_dir, 'test_split.csv')

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Data splits successfully generated and saved to Drive!")
print(f"Train set: {len(train_df)} samples -> {train_csv_path}")
print(f"Test set:  {len(test_df)} samples -> {test_csv_path}")
print("\nSample of train_split.csv (Notice the 3 strict columns):")
print(train_df.head())