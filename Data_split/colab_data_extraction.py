# 1. Mount Google Drive
from google.colab import drive
import os
import zipfile

drive.mount('/content/drive')

# 2. Define the path to your zip file
zip_path = '/content/drive/MyDrive/IIIT_project/TESS_dataset/archive.zip'
extract_path = '/content/tess_dataset'

# 3. Unzip the dataset
print("Unzipping dataset... this might take a minute.")
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete!")
else:
    print("Dataset already extracted in this session.")

# 4. Verify extraction by listing what's inside
try:
    extracted_contents = os.listdir(extract_path)
    print(f"\nContents of extraction folder: {extracted_contents[:10]}")

    # Let's peek inside the first folder to see the audio files
    if len(extracted_contents) > 0:
        first_item_path = os.path.join(extract_path, extracted_contents[0])
        if os.path.isdir(first_item_path):
            print(f"Contents of '{extracted_contents[0]}': {os.listdir(first_item_path)[:5]}")
except Exception as e:
    print(f"Error checking contents: {e}")