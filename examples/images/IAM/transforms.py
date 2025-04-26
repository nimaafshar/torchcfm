import numpy as np
import torch
import PIL

# Create a transform class to standardize aspect ratio
class StandardizeAspectRatio:
    def __init__(self, target_aspect_ratio, background_value=1.0):
        self.target_aspect_ratio = target_aspect_ratio
        self.background_value = background_value
        
    def __call__(self, img):
        # Convert PIL image to numpy array and normalize
        if isinstance(img, PIL.Image.Image):
            img = np.array(img).astype(np.float32) / 255.0
        
        h, w = img.shape
        current_aspect = w / h
        
        if current_aspect < self.target_aspect_ratio:
            # Image is too tall, add width
            new_width = int(h * self.target_aspect_ratio)
            new_img = np.full((h, new_width), self.background_value)
            # Center the original image
            start_x = (new_width - w) // 2
            new_img[:, start_x:start_x+w] = img
        else:
            # Image is too wide, add height
            new_height = int(w / self.target_aspect_ratio)
            new_img = np.full((new_height, w), self.background_value)
            # Center the original image
            start_y = (new_height - h) // 2
            new_img[start_y:start_y+h, :] = img
            
        return torch.from_numpy(new_img).unsqueeze(0)


