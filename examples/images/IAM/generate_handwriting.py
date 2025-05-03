import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from absl import app, flags
from PIL import Image
import numpy as np

from iam_dataset import IAMWordDataset
from conditional_unet import CharacterConditionedUNet
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

# Define flags
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("model_channels", 128, help="base channel dimension of UNet")
flags.DEFINE_integer("image_channels", 1, help="Number of channels in the input image")
flags.DEFINE_integer("num_res_blocks", 2, help="number of residual blocks")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_integer("num_heads", 4, help="number of attention heads")
flags.DEFINE_integer("num_head_channels", 64, help="number of channels per attention head")
flags.DEFINE_float("dropout", 0.1, help="dropout rate")
flags.DEFINE_integer("embed_dim", 64, help="character embedding dimension")
flags.DEFINE_string("attention_resolutions", "16", help="UNet attention resolutions")

# Generation parameters
flags.DEFINE_string("checkpoint", "", help="path to model checkpoint")
flags.DEFINE_string("data_dir", "~/datasets/IAM", help="IAM dataset directory")
flags.DEFINE_integer("image_height", 64, help="image height")
flags.DEFINE_integer("image_width", 128, help="image width")
flags.DEFINE_integer("num_samples", 16, help="number of samples to generate")
flags.DEFINE_string("output_dir", "results/generated", help="output directory")
flags.DEFINE_string("text", "", help="text to generate (if empty, will use random words from dataset)")
flags.DEFINE_integer("num_inference_steps", 50, help="number of inference steps")
flags.DEFINE_integer("max_word_length", 6, help="maximum word length")
flags.DEFINE_integer("min_word_length", 4, help="minimum word length")

# Create FM variable but initialize it later
FM = None

def encode_text(text, dataset):
    """Encode text to character indices."""
    indices = []
    for char in text:
        if char in dataset.char_to_idx:
            indices.append(dataset.char_to_idx[char])
        else:
            # Use a fallback character (space or the first character in the vocab)
            if ' ' in dataset.char_to_idx:
                indices.append(dataset.char_to_idx[' '])
            else:
                indices.append(0)  # First character as fallback
    return torch.tensor(indices, dtype=torch.long)

def generate_samples(model, dataset, device, text=None, num_samples=1):
    """Generate handwriting samples from the model."""
    model.eval()
    with torch.no_grad():
        if text:
            # Encode the provided text
            char_indices = encode_text(text, dataset)
            char_indices = char_indices.unsqueeze(0).repeat(num_samples, 1).to(device)
            texts = [text] * num_samples
        else:
            # Sample random words from the dataset
            indices = torch.randint(0, len(dataset), (num_samples,))
            char_indices = []
            texts = []
            for idx in indices:
                _, text = dataset[idx]
                char_indices.append(text)
                texts.append(dataset.decode_text(text))
            
            # Pad character sequences to the same length
            max_len = max(len(text) for text in char_indices)
            padded_indices = torch.zeros((num_samples, max_len), dtype=torch.long)
            for i, text in enumerate(char_indices):
                padded_indices[i, :len(text)] = text
            
            # Move to device
            char_indices = padded_indices.to(device)
        
        # Generate samples
        x0 = torch.randn(num_samples, FLAGS.image_channels, FLAGS.image_height, FLAGS.image_width).to(device)
        
        # Generate samples exactly as in train_handwriting.py
        samples = []
        for i in range(num_samples):
            # Sample a single image with t=1 (initial timestep)
            t = torch.ones(1, 1).to(device)
            xt = x0[i:i+1].clone()
            
            # Integrate the flow with multiple steps (like in train_handwriting.py)
            dt = 1.0 / FLAGS.num_inference_steps
            for j in range(FLAGS.num_inference_steps):
                t_j = t * (1.0 - j * dt)
                vt = model(t_j, xt, char_indices[i:i+1])
                xt = xt + vt * dt
            
            samples.append(xt)
        
        # Stack samples
        samples = torch.cat(samples, dim=0)
        
        # Denormalize
        samples = (samples + 1) / 2
        
        return samples, texts

def load_vocab_from_checkpoint(checkpoint_path, device):
    """Extract vocabulary size from the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "ema_model" in checkpoint:
        state_dict = checkpoint["ema_model"]
    else:
        state_dict = checkpoint["model"]
    
    # Get vocabulary size from char_embedding weight shape
    if "char_embedding.embedding.weight" in state_dict:
        vocab_size = state_dict["char_embedding.embedding.weight"].shape[0]
        return vocab_size
    return None

def main(argv):
    """Generate handwriting samples from a trained model."""
    global FM
    
    # Initialize FM after flags are parsed
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    else:
        FM = ConditionalFlowMatcher(sigma=0.0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset with the exact parameters from training
    # These values must match what was used during training
    dataset = IAMWordDataset(
        root_dir=FLAGS.data_dir,
        split='train',
        transform=None,
        max_word_length=FLAGS.max_word_length,  # Use the same max_word_length as training
        min_word_length=FLAGS.min_word_length,  # Use the same min_word_length as training
    )
    
    # Parse attention_resolutions from string flag to tuple of ints
    try:
        attention_resolutions_int = tuple(int(res) for res in FLAGS.attention_resolutions.split(",") if res)
    except ValueError:
        print(f"Warning: Invalid attention_resolutions string '{FLAGS.attention_resolutions}'. Using empty tuple.")
        attention_resolutions_int = tuple()
    
    # Check if we need to extract vocab size from checkpoint
    vocab_size = None
    if FLAGS.checkpoint:
        vocab_size = load_vocab_from_checkpoint(FLAGS.checkpoint, device)
    
    if vocab_size is None:
        vocab_size = len(dataset.char_to_idx)
        print(f"Using dataset vocabulary size: {vocab_size}")
    else:
        print(f"Using checkpoint vocabulary size: {vocab_size}")
    
    # Create model
    model = CharacterConditionedUNet(
        image_size=FLAGS.image_width, 
        image_channels=FLAGS.image_channels, 
        model_channels=FLAGS.model_channels,
        num_res_blocks=FLAGS.num_res_blocks,
        channel_mult=FLAGS.channel_mult,
        embed_dim=FLAGS.embed_dim, 
        vocab_size=vocab_size,  # Use the vocabulary size from the checkpoint
        attention_resolutions=attention_resolutions_int,
        dropout=FLAGS.dropout,
        learn_sigma=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
        num_heads=FLAGS.num_heads,
        num_head_channels=FLAGS.num_head_channels,
    ).to(device)
    
    # Load checkpoint
    if FLAGS.checkpoint:
        checkpoint = torch.load(FLAGS.checkpoint, map_location=device)
        if "ema_model" in checkpoint:
            model.load_state_dict(checkpoint["ema_model"])
            print(f"Loaded EMA model from {FLAGS.checkpoint}")
        else:
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded model from {FLAGS.checkpoint}")
    else:
        print("No checkpoint provided. Using randomly initialized model.")
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Generate samples
    samples, texts = generate_samples(
        model, dataset, device, FLAGS.text, FLAGS.num_samples
    )
    
    # Save samples
    for i, (sample, text) in enumerate(zip(samples, texts)):
        plt.figure(figsize=(8, 2))
        plt.imshow(sample.squeeze().cpu().numpy(), cmap='gray')
        plt.title(text)
        plt.axis('off')
        plt.savefig(os.path.join(FLAGS.output_dir, f"sample_{i}.png"))
        plt.close()
        
        # Also save as numpy array
        np.save(
            os.path.join(FLAGS.output_dir, f"sample_{i}.npy"),
            sample.squeeze().cpu().numpy()
        )
    
    print(f"Generated {FLAGS.num_samples} samples in {FLAGS.output_dir}")

if __name__ == "__main__":
    app.run(main)
