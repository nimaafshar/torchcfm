import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from absl import app, flags
from PIL import Image
import numpy as np
import copy

from iam_dataset import IAMWordDataset
from transforms import StandardizeAspectRatio
from conditional_unet import CharacterConditionedUNet
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
# Import odeint for sampling from torchdiffeq
from torchdiffeq import odeint 

# Define flags
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("model_channels", 128, help="base channel dimension of UNet")
flags.DEFINE_integer("image_channels", 1, help="Number of channels in the input image (e.g., 1 for grayscale)")
flags.DEFINE_integer("num_res_blocks", 2, help="number of residual blocks")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_string("attention_resolutions", "16", help="UNet attention resolutions (comma-separated string, e.g., '32,16,8')")
flags.DEFINE_integer("num_heads", 1, help="Number of attention heads")
flags.DEFINE_integer("num_head_channels", -1, help="Number of channels per attention head (-1 to use num_heads)")
flags.DEFINE_float("dropout", 0.1, help="dropout rate")
flags.DEFINE_integer("embed_dim", 64, help="character embedding dimension (used for conditioning channels)")

# Training parameters
flags.DEFINE_float("lr", 2e-4, help="learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 10000, help="total training steps")
flags.DEFINE_integer("warmup", 2000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Dataset parameters
flags.DEFINE_string("data_dir", "~/datasets/IAM", help="IAM dataset directory")
flags.DEFINE_integer("max_word_length", 5, help="maximum word length")
flags.DEFINE_integer("min_word_length", 2, help="minimum word length")
flags.DEFINE_integer("image_height", 64, help="image height")
flags.DEFINE_integer("image_width", 128, help="image width")

# Evaluation parameters
flags.DEFINE_integer("save_step", 2000, help="frequency of saving checkpoints")
flags.DEFINE_integer("eval_step", 1000, help="frequency of evaluation")
flags.DEFINE_integer("num_samples", 16, help="number of samples to generate during evaluation")
flags.DEFINE_integer("num_inference_steps", 50, help="number of steps for inference")

# GPU selection parameter
flags.DEFINE_integer("gpu", 0, help="GPU ID to use (0 for first GPU, 1 for second GPU, etc.)")

# Create FM variable but initialize it in train function
FM = None

def ema(model, ema_model, decay):
    """Update EMA model parameters."""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def warmup_lr(step):
    """Learning rate warmup schedule."""
    return min(step, FLAGS.warmup) / FLAGS.warmup

def generate_samples(model, dataset, device, num_samples, savedir, step, net_="normal"):
    """Generate and save samples from the model."""
    model.eval()
    with torch.no_grad():
        # Sample random words from the dataset
        indices = torch.randint(0, len(dataset), (num_samples,))
        char_indices = []
        for idx in indices:
            _, text = dataset[idx]
            char_indices.append(text)
        
        # Pad character sequences to the same length
        max_len = max(len(text) for text in char_indices)
        padded_indices = torch.zeros((num_samples, max_len), dtype=torch.long)
        for i, text in enumerate(char_indices):
            padded_indices[i, :len(text)] = text
        
        # Move to device
        char_indices = padded_indices.to(device)
        
        # Generate samples
        x0 = torch.randn(num_samples, FLAGS.image_channels, FLAGS.image_height, FLAGS.image_width).to(device)
        
        # Generate samples
        samples = []
        for i in range(num_samples):
            # Start with noise and set initial timestep to t=1
            t = torch.ones(1, 1).to(device)
            xt = x0[i:i+1].clone()
            
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
        
        # Save samples
        os.makedirs(savedir, exist_ok=True)
        for i in range(num_samples):
            plt.figure(figsize=(4, 1))
            plt.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(savedir, f"sample_{step}_{i}_{net_}.png"))
            plt.close()
    
    model.train()

def train(argv):
    """Train the handwriting generation model."""
    global FM
    
    # Initialize FM after flags are parsed
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    else:
        FM = ConditionalFlowMatcher(sigma=0.0)
    
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )
    
    # Set GPU device - add this to select the GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Using GPU: {FLAGS.gpu}")
    
    # Set device
    device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Create dataset and dataloader
    transform = transforms.Compose([
        StandardizeAspectRatio(2.0),  # Standardize aspect ratio to 2:1
        transforms.ToPILImage(),
        transforms.Resize((FLAGS.image_height, FLAGS.image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    dataset = IAMWordDataset(
        root_dir=FLAGS.data_dir,
        split='train',
        transform=transform,
        max_word_length=FLAGS.max_word_length,
        min_word_length=FLAGS.min_word_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    
    # Parse attention_resolutions from string flag to tuple of ints
    try:
        attention_resolutions_int = tuple(int(res) for res in FLAGS.attention_resolutions.split(",") if res)
    except ValueError:
        print(f"Warning: Invalid attention_resolutions string '{FLAGS.attention_resolutions}'. Using empty tuple.")
        attention_resolutions_int = tuple()
    
    # Create model - passing parameters directly to UNetModel now
    model = CharacterConditionedUNet(
        image_size=FLAGS.image_width, # Base UNetModel uses image_size
        image_channels=FLAGS.image_channels, 
        model_channels=FLAGS.model_channels, # Base channel dim
        num_res_blocks=FLAGS.num_res_blocks,
        channel_mult=FLAGS.channel_mult, # Pass the list directly
        embed_dim=FLAGS.embed_dim, 
        vocab_size=len(dataset.char_to_idx),
        attention_resolutions=attention_resolutions_int, # Pass the tuple of ints
        dropout=FLAGS.dropout,
        learn_sigma=False, # Assuming not learning sigma
        use_checkpoint=False, # Default
        use_scale_shift_norm=False, # Default
        resblock_updown=False, # Default
        use_fp16=False, # Default
        num_heads=FLAGS.num_heads,
        num_head_channels=FLAGS.num_head_channels,
        # num_heads_upsample=-1, # Default in UNetModel
        # use_new_attention_order=False, # Default in UNetModel
    ).to(device)
    
    # Use copy.deepcopy like in CIFAR10
    ema_model = copy.deepcopy(model)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    
    # Training loop
    savedir = f"results/{FLAGS.model}/"
    os.makedirs(savedir, exist_ok=True)
    
    step = 0
    with tqdm(total=FLAGS.total_steps) as pbar:
        while step < FLAGS.total_steps:
            for batch in dataloader:
                if step >= FLAGS.total_steps:
                    break
                
                # Get batch
                images, char_indices = batch
                images = images.to(device)
                char_indices = char_indices.to(device)
                
                # Sample noise and time
                x0 = torch.randn_like(images)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, images)
                
                # Forward pass
                vt = model(t, xt, char_indices)
                
                # Compute loss
                loss = torch.mean((vt - ut) ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip)
                optimizer.step()
                scheduler.step()
                
                # Update EMA model
                ema(model, ema_model, FLAGS.ema_decay)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
                # Generate samples
                if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                    generate_samples(model, dataset, device, FLAGS.num_samples, savedir, step, "normal")
                    generate_samples(ema_model, dataset, device, FLAGS.num_samples, savedir, step, "ema")
                    
                    # Save checkpoint
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                        },
                        os.path.join(savedir, f"{FLAGS.model}_handwriting_weights_step_{step}.pt"),
                    )
                
                # Add before sampling
                # params_equal = True
                # for p1, p2 in zip(model.parameters(), ema_model.parameters()):
                #     if not torch.allclose(p1, p2, rtol=1e-3):
                #         params_equal = False
                #         break
                # print(f"Model and EMA parameters identical: {params_equal}")
                
                step += 1

if __name__ == "__main__":
    app.run(train) 