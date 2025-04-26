import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import re

class IAMWordDataset(Dataset):
    """Dataset class for IAM handwritten word images.
    
    The IAM dataset consists of handwritten word images and their transcriptions.
    The data is organized as follows:
    
    1. Images are stored in the 'words' directory, organized by form ID:
       words/
         a01/
           a01-000u-02-06.png
           a01-000u-02-07.png
           ...
         a02/
           ...
    
    2. Ground truth is stored in 'ascii/words.txt' with the following format:
       word_id writer_id line_id word_transcription [additional_metadata]
       Example: a01-000u-02-06 000u 02 meeting
    
    The word_id format is: {form_id}-{writer_id}-{line_id}-{word_id}
    - form_id: Form identifier (e.g., a01)
    - writer_id: Writer identifier (e.g., 000u)
    - line_id: Line number in the form
    - word_id: Word number within the line
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        max_word_length: int = 32,
        min_word_length: int = 1
    ):
        """
        Initialize the IAM word dataset.
        
        Args:
            root_dir: Root directory of the IAM dataset
            split: One of ['train', 'val', 'test']
            transform: Optional transform to be applied on images
            max_word_length: Maximum word length to consider
            min_word_length: Minimum word length to consider
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_word_length = max_word_length
        self.min_word_length = min_word_length
        
        # Define the character set (including special tokens)
        self.char_to_idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        # Load the words.txt file
        self.words_file = os.path.join(root_dir, 'ascii', 'words.txt')
        self.words_dir = os.path.join(root_dir, 'words')
        
        # Load and filter the dataset
        self.samples = self._load_dataset()
        
        # Build character vocabulary
        self._build_vocabulary()
        
    def _load_dataset(self) -> List[Tuple[str, str]]:
        """Load and filter the dataset based on split and word length constraints."""
        samples = []
        
        # Read the words.txt file
        with open(self.words_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                # Parse the line
                parts = line.strip().split()
                if len(parts) < 9:  # Skip malformed lines
                    continue
                    
                word_id = parts[0]
                transcription = parts[-1]
                
                # Skip words that are too short or too long
                if not (self.min_word_length <= len(transcription) <= self.max_word_length):
                    continue
                
                # Check if the image file exists
                word_attributes = word_id.split('-')
                form_id = word_attributes[0]
                writer_id = word_attributes[1]
                
                image_path = os.path.join(self.words_dir, form_id, f"{form_id}-{writer_id}",f"{word_id}.png")
                
                if not os.path.exists(image_path):
                    print(f"Image file does not exist: {image_path}")
                    continue
                
                # Determine split based on form ID (you may want to adjust this)
                form_num = int(form_id[1:])
                
                if self.split == 'train' and form_num <= 600:
                    samples.append((image_path, transcription))
                elif self.split == 'val' and 600 < form_num <= 700:
                    samples.append((image_path, transcription))
                elif self.split == 'test' and form_num > 700:
                    samples.append((image_path, transcription))
        
        return samples
    
    def _build_vocabulary(self):
        """Build character vocabulary from the dataset."""
        # Add all unique characters to vocabulary
        for _, transcription in self.samples:
            for char in transcription:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = len(self.char_to_idx)
        
        # Create reverse mapping
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Convert text to tensor of character indices."""
        indices = [self.char_to_idx['<sos>']]
        for char in text:
            indices.append(self.char_to_idx.get(char, self.char_to_idx['<unk>']))
        indices.append(self.char_to_idx['<eos>'])
        
        # Pad to max length
        indices.extend([self.char_to_idx['<pad>']] * (self.max_word_length + 2 - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, transcription = self.samples[idx]
        
        # Load and preprocess image
        with Image.open(image_path) as raw_image:
            image = raw_image.convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.unsqueeze(0)  # Add channel dimension
        
        # Encode text
        text = self._encode_text(transcription)
        
        return image, text
    
    def decode_text(self, indices: torch.Tensor) -> str:
        """Convert tensor of indices back to text."""
        text = []
        for tensor_idx in indices:
            idx = tensor_idx.item()
            if idx == self.char_to_idx['<eos>']:
                break
            if idx not in [self.char_to_idx['<sos>'], self.char_to_idx['<pad>']]:
                text.append(self.idx_to_char[idx])
        return ''.join(text)

