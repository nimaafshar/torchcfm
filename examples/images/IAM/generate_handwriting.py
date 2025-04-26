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
flags.DEFINE_integer("num_channels", 128, help="base channel of UNet")
flags.DEFINE_integer("num_res_blocks", 2, help="number of residual blocks")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_integer("num_heads", 4, help="number of attention heads")
flags.DEFINE_integer("num_head_channels", 64, help="number of channels per attention head")
flags.DEFINE_float("dropout", 0.1, help="dropout rate")
flags.DEFINE_integer("embed_dim", 64, help="character embedding dimension")

# Generation parameters
flags.DEFINE_string("checkpoint", "", help="path to model checkpoint")
flags.DEFINE_string("data_dir", "~/datasets/IAM", help="IAM dataset directory")
flags.DEFINE_integer("image_size", 32, help="image height")
flags.DEFINE_integer("image_width", 128, help="image width")
flags.DEFINE_integer("num_samples", 16, help="number of samples to generate")
flags.DEFINE_string("output_dir", "results/generated", help="output directory")
flags.DEFINE_string("text", "", help="text to generate (if empty, will use random words from dataset)")
flags.DEFINE_integer("num_inference_steps", 50, help="number of inference steps")

def encode_text(text, dataset):
    """Encode text to character indices."""
    indices = []
    for char in text:
        if char in dataset.char_to_idx:
            indices.append(dataset.char_to_idx[char])
        else:
            indices.append(dataset.char_to_idx[''])
    return torch.tensor(indices, dtype=torch.long)

def generate_samples(model, dataset, device, text=None, num_samples=1):
    """Generate handwriting samples from the model."""
    model.eval()
    with torch.no_grad():
        if text:
            # Encode the provided text
            char_indices = encode_text(text, dataset)
            char_indices = char_indices.unsqueeze(0).repeat(num_samples, 1).to(device)
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
        x0 = torch.randn(num_samples, 1, FLAGS.image_size, FLAGS.image_width).to(device)
        
        # Sample from the model
        if FLAGS.model == "otcfm":
            fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        else:
            fm = ConditionalFlowMatcher(sigma=0.0)
        
        # Generate samples
        samples = []
        for i in range(num_samples):
            # Sample a single image
            t, xt, ut = fm.sample_location_and_conditional_flow(x0[i:i+1], None)
            
            # Integrate the flow with multiple steps
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
        
        return samples, texts if not text else [text] * num_samples

def main(argv):
    """Generate handwriting samples from a trained model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = IAMWordDataset(
        root_dir=FLAGS.data_dir,
        split='train',
        transform=None,
        max_word_length=32,
        min_word_length=1,
    )
    
    # Create model
    model = CharacterConditionedUNet(
        dim=(1, FLAGS.image_size, FLAGS.image_width),
        num_channels=FLAGS.num_channels,
        num_res_blocks=FLAGS.num_res_blocks,
        channel_mult=[int(x) for x in FLAGS.channel_mult],
        num_heads=FLAGS.num_heads,
        num_head_channels=FLAGS.num_head_channels,
        dropout=FLAGS.dropout,
        vocab_size=len(dataset.char_to_idx),
        embed_dim=FLAGS.embed_dim,
    ).to(device)
    
    # Load checkpoint
    if FLAGS.checkpoint:
        checkpoint = torch.load(FLAGS.checkpoint, map_location=device)
        if "ema_model" in checkpoint:
            model.load_state_dict(checkpoint["ema_model"])
        else:
            model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {FLAGS.checkpoint}")
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
