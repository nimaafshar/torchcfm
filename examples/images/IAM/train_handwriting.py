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
import io
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from sklearn.metrics.pairwise import cosine_similarity
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import json
import datetime

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
flags.DEFINE_float("lr_decay_rate", 0.9, help="learning rate decay rate (applied every lr_decay_steps)")
flags.DEFINE_integer("lr_decay_steps", 2000, help="number of steps between learning rate decay")
flags.DEFINE_float("min_lr", 1e-6, help="minimum learning rate")
flags.DEFINE_float("l2_reg_weight", 0.001, help="weight for L2 regularization of velocity field")
flags.DEFINE_boolean("use_weighted_loss", True, help="whether to use weighted loss based on time")
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
flags.DEFINE_integer("save_step", 5000, help="frequency of saving checkpoints")
flags.DEFINE_integer("eval_step", 1000, help="frequency of evaluation")
flags.DEFINE_integer("log_step", 100, help="frequency of logging to tensorboard")
flags.DEFINE_integer("num_samples", 16, help="number of samples to generate during evaluation")
flags.DEFINE_integer("num_inference_steps", 50, help="number of steps for inference")
flags.DEFINE_string("logdir", "logs", help="directory for tensorboard logs")
flags.DEFINE_string("run_name", "default_run", help="name for this run in tensorboard")

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

def lr_schedule(step):
    """Learning rate schedule with warmup and decay.
    First warms up the learning rate, then decays it exponentially."""
    if step < FLAGS.warmup:
        # Warmup phase
        return step / FLAGS.warmup
    else:
        # Decay phase - exponential decay after warmup
        decay_factor = FLAGS.lr_decay_rate ** ((step - FLAGS.warmup) / FLAGS.lr_decay_steps)
        # Apply minimum learning rate
        return max(decay_factor, FLAGS.min_lr / FLAGS.lr)

def tensor_to_pil(images):
    """Convert batch of tensors to PIL Images for FID/KID calculation."""
    # Denormalize from [-1, 1] to [0, 255]
    images = (images + 1) / 2 * 255
    # Convert to uint8
    images = images.to(torch.uint8)
    return images

def calculate_metrics(real_images, generated_images, device):
    """Calculate FID and KID between real and generated images."""
    # Get batch size
    batch_size = min(real_images.size(0), generated_images.size(0))
    
    # Ensure we have enough images for metrics calculation
    if batch_size < 2:
        print("Batch size too small for metrics calculation")
        return 0.0, 0.0, 0.0
    
    # Set subset size to be smaller than batch size
    subset_size = max(2, batch_size // 2)  # Use at least 2, or half the batch size
    
    # Ensure images are in the right format: NCHW, uint8, [0, 255]
    real_images_uint8 = tensor_to_pil(real_images)
    generated_images_uint8 = tensor_to_pil(generated_images)
    
    # If images are grayscale, convert to RGB by repeating channels
    if real_images_uint8.size(1) == 1:
        real_images_uint8 = real_images_uint8.repeat(1, 3, 1, 1)
        generated_images_uint8 = generated_images_uint8.repeat(1, 3, 1, 1)
    
    # Initialize metrics with smaller subset size
    fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=64, normalize=True).to(device)
    
    # Update metrics with real and fake images
    fid.update(real_images_uint8, real=True)
    fid.update(generated_images_uint8, real=False)
    
    kid.update(real_images_uint8, real=True)
    kid.update(generated_images_uint8, real=False)
    
    # Calculate and return metrics
    try:
        fid_score = fid.compute()
        kid_score, kid_std = kid.compute()
        return fid_score.item(), kid_score.item(), kid_std.item()
    except Exception as e:
        print(f"Error in metric computation: {e}")
        return 0.0, 0.0, 0.0

def generate_samples_for_tensorboard(model, dataset, device, num_samples, writer, step, model_type="normal"):
    """Generate samples and log to tensorboard."""
    model.eval()
    with torch.no_grad():
        # Sample random words from the dataset
        indices = torch.randint(0, len(dataset), (num_samples,))
        
        # Get images and texts
        real_images = []
        char_indices = []
        words = []
        
        for idx in indices:
            image, text = dataset[idx]
            real_images.append(image)
            char_indices.append(text)
            # Convert char indices to actual word for logging
            word = dataset.get_word_from_indices(text)
            words.append(word)
        
        real_images = torch.stack(real_images).to(device)
        
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
        
        # Denormalize for visualization but keep in [-1, 1] for metrics
        samples_normalized = (samples + 1) / 2
        
        # Log the images to tensorboard with their corresponding text
        grid_real = vutils.make_grid(real_images, nrow=4, normalize=True)
        grid_generated = vutils.make_grid(samples_normalized, nrow=4, normalize=True)
        
        writer.add_image(f'Real_Images', grid_real, step)
        writer.add_image(f'Generated_Images/{model_type}', grid_generated, step)
        
        # Add side-by-side comparison with text labels
        fig, axs = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
        for i in range(num_samples):
            axs[i, 0].imshow(real_images[i].squeeze().cpu().numpy(), cmap='gray')
            axs[i, 0].set_title(f"Real: {words[i]}")
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(samples_normalized[i].squeeze().cpu().numpy(), cmap='gray')
            axs[i, 1].set_title(f"Generated: {words[i]}")
            axs[i, 1].axis('off')
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Add figure to tensorboard
        image = Image.open(buf)
        image = transforms.ToTensor()(image)
        writer.add_image(f'Comparison/{model_type}', image, step)
        
        # Calculate and log metrics
        try:
            fid_score, kid_score, kid_std = calculate_metrics(real_images, samples, device)
            writer.add_scalar(f'Metrics/{model_type}/FID', fid_score, step)
            writer.add_scalar(f'Metrics/{model_type}/KID', kid_score, step)
            writer.add_scalar(f'Metrics/{model_type}/KID_std', kid_std, step)
            print(f"{model_type} FID: {fid_score:.4f}, KID: {kid_score:.4f} ± {kid_std:.4f}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    model.train()
    return samples

def generate_samples(model, dataset, device, num_samples, savedir, step, net_="normal"):
    """Generate and save samples from the model."""
    model.eval()
    with torch.no_grad():
        # Sample random words from the dataset
        indices = torch.randint(0, len(dataset), (num_samples,))
        char_indices = []
        words = []
        for idx in indices:
            _, text = dataset[idx]
            char_indices.append(text)
            # Convert char indices to actual word for saving
            word = dataset.get_word_from_indices(text)
            words.append(word)
        
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
            plt.title(words[i])
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
        "lr, min_lr, total_steps, warmup, lr_decay_rate, lr_decay_steps, ema_decay, save_step:",
        FLAGS.lr,
        FLAGS.min_lr,
        FLAGS.total_steps,
        FLAGS.warmup,
        FLAGS.lr_decay_rate,
        FLAGS.lr_decay_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )
    
    print(
        "Loss config - L2 reg weight: {}, Use weighted loss: {}".format(
            FLAGS.l2_reg_weight, FLAGS.use_weighted_loss
        )
    )
    
    # Set GPU device - add this to select the GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Using GPU: {FLAGS.gpu}")
    
    # Set device
    device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize TensorBoard writer
    logdir = os.path.join(FLAGS.logdir, FLAGS.model, FLAGS.run_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    # Save run configuration
    config = {
        "model": FLAGS.model,
        "run_name": FLAGS.run_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "lr": FLAGS.lr,
        "min_lr": FLAGS.min_lr,
        "total_steps": FLAGS.total_steps,
        "warmup": FLAGS.warmup,
        "lr_decay_rate": FLAGS.lr_decay_rate,
        "lr_decay_steps": FLAGS.lr_decay_steps,
        "batch_size": FLAGS.batch_size,
        "ema_decay": FLAGS.ema_decay,
        "model_channels": FLAGS.model_channels,
        "image_channels": FLAGS.image_channels,
        "num_res_blocks": FLAGS.num_res_blocks,
        "channel_mult": FLAGS.channel_mult,
        "attention_resolutions": FLAGS.attention_resolutions,
        "num_heads": FLAGS.num_heads,
        "embed_dim": FLAGS.embed_dim,
        "dropout": FLAGS.dropout,
        "max_word_length": FLAGS.max_word_length,
        "min_word_length": FLAGS.min_word_length,
        "image_height": FLAGS.image_height,
        "image_width": FLAGS.image_width,
        "l2_reg_weight": FLAGS.l2_reg_weight,
        "use_weighted_loss": FLAGS.use_weighted_loss,
        "grad_clip": FLAGS.grad_clip,
    }
    
    # Save config to TensorBoard log directory and results directory
    with open(os.path.join(logdir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
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
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Training loop
    savedir = f"results/{FLAGS.model}/{FLAGS.run_name}/"
    os.makedirs(savedir, exist_ok=True)
    
    # Also save config to results directory
    with open(os.path.join(savedir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    step = 0
    
    # Create LR tracker for tensorboard
    def get_lr():
        return optimizer.param_groups[0]['lr']
    
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
                
                # Compute loss with regularization
                # Basic MSE loss
                mse_loss = torch.mean((vt - ut) ** 2)
                
                # Add L2 regularization for the velocity field
                l2_reg = torch.mean(vt**2) * FLAGS.l2_reg_weight
                
                # Add per-sample loss normalization if enabled
                weighted_mse = 0.0
                if FLAGS.use_weighted_loss:
                    batch_weights = 1.0 / (torch.sum(t * (1-t), dim=1) + 1e-5)
                    batch_weights = batch_weights / batch_weights.mean()
                    weighted_mse = torch.mean(batch_weights * torch.mean((vt - ut) ** 2, dim=[1,2,3]))
                    loss = weighted_mse + l2_reg
                else:
                    loss = mse_loss + l2_reg
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Advanced gradient clipping per parameter
                # This helps with training stability for complex models
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-FLAGS.grad_clip, FLAGS.grad_clip)
                
                optimizer.step()
                scheduler.step()
                
                # Update EMA model
                ema(model, ema_model, FLAGS.ema_decay)
                
                # Log to tensorboard
                if step % FLAGS.log_step == 0:
                    writer.add_scalar('Loss/train', loss.item(), step)
                    writer.add_scalar('Loss/mse', mse_loss.item(), step)
                    writer.add_scalar('Loss/l2_reg', l2_reg.item(), step)
                    writer.add_scalar('Loss/weighted_mse', weighted_mse.item(), step)
                    writer.add_scalar('Learning_Rate', get_lr(), step)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), mse=mse_loss.item())
                
                # Generate samples and log metrics
                if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                    # Generate samples for visualization in TensorBoard
                    generate_samples_for_tensorboard(model, dataset, device, FLAGS.num_samples, writer, step, "normal")
                    generate_samples_for_tensorboard(ema_model, dataset, device, FLAGS.num_samples, writer, step, "ema")
                
                # Save checkpoint
                if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                    # Generate samples to disk
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
                        os.path.join(savedir, f"{FLAGS.model}_{FLAGS.run_name}_handwriting_weights_step_{step}.pt"),
                    )
                
                step += 1
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    app.run(train) 