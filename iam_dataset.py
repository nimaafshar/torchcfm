import os
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class IAMDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_height=32):
        """
        Args:
            root_dir (str): Directory with the IAM dataset
            transform (callable, optional): Optional transform to be applied on a sample
            target_height (int): Target height for resizing images while maintaining aspect ratio
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_height = target_height
        
        # Read the words.txt file from the ascii directory
        words_file = os.path.join(root_dir, 'ascii', 'words.txt')
        
        # Parse the words.txt file manually since it has a specific format
        self.annotations = []
        with open(words_file, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Parse the line
                parts = line.strip().split()
                if len(parts) >= 9:  # Ensure we have enough parts
                    word_id = parts[0]
                    result = parts[1]
                    graylevel = int(parts[2])
                    x = int(parts[3])
                    y = int(parts[4])
                    width = int(parts[5])
                    height = int(parts[6])
                    tag = parts[7]
                    transcription = parts[8]
                    
                    # Only include words with 'ok' segmentation result
                    if result == 'ok':
                        self.annotations.append({
                            'word_id': word_id,
                            'transcription': transcription,
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'tag': tag
                        })
        
        # Convert to DataFrame for easier access
        self.annotations = pd.DataFrame(self.annotations)
        
        # Create base transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get word information
        word_info = self.annotations.iloc[idx]
        word_id = word_info['word_id']
        transcription = word_info['transcription']
        
        # Construct path to word image based on the specific format
        # Format: words/a01/a01-000u/a01-000u-00-00.png
        form_id = word_id.split('-')[0] + '-' + word_id.split('-')[1]
        word_path = os.path.join(self.root_dir, 'words', form_id.lower(), 
                               word_id.lower(), f"{word_id.lower()}.png")
        
        # Load and preprocess image
        image = Image.open(word_path).convert('L')  # Convert to grayscale
        
        # Calculate new width maintaining aspect ratio
        width, height = image.size
        new_width = int(width * (self.target_height / height))
        
        # Resize image
        image = image.resize((new_width, self.target_height), Image.Resampling.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'text': transcription,
            'word_id': word_id
        } 