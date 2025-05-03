import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import subprocess
from tqdm import tqdm

def generate_sample(checkpoint_path, output_dir, text="hello", image_height=64, image_width=128, use_ema=True):
    """Generate a sample using the given checkpoint."""
    cmd = [
        "python3", "examples/images/IAM/generate_handwriting.py",
        f"--checkpoint={checkpoint_path}",
        f"--data_dir=/mnt/speech6/bagher/nafshar/datasets/IAM",
        f"--image_height={image_height}",
        f"--image_width={image_width}",
        f"--num_samples=1",
        f"--output_dir={output_dir}",
        f"--model_channels=128",
        f"--text={text}"
    ]
    
    # Use environment variable to control which model to use (EMA or regular)
    env = os.environ.copy()
    if not use_ema:
        env["USE_REGULAR_MODEL"] = "1"
    
    # Run the command and capture output
    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env)
        # Return the path to the first generated sample
        return os.path.join(output_dir, "sample_0.png")
    except subprocess.CalledProcessError as e:
        print(f"Error generating sample from {checkpoint_path}:")
        print(e.stderr.decode())
        return None

def main():
    parser = argparse.ArgumentParser(description='Compare handwriting samples from different checkpoints')
    parser.add_argument('--steps', type=str, default="0,4000,8000,12000,16000,20000,24000,28000", 
                        help='Comma-separated list of steps to compare')
    parser.add_argument('--texts', type=str, default="hello,world,flow", 
                        help='Comma-separated list of texts to generate')
    parser.add_argument('--compare_models', action='store_true', 
                        help='Compare regular vs EMA models')
    args = parser.parse_args()
    
    # Parse arguments
    selected_steps = [int(step) for step in args.steps.split(',')]
    texts = args.texts.split(',')
    
    # Create directory for comparison results
    comparison_dir = "results/checkpoint_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get list of available checkpoints
    checkpoint_dir = "results/otcfm"
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
    
    # Extract step numbers for sorting
    checkpoint_steps = []
    for f in checkpoint_files:
        try:
            step = int(f.split("_step_")[1].split(".")[0])
            checkpoint_steps.append((step, f))
        except (IndexError, ValueError):
            continue
    
    # Sort checkpoints by step
    checkpoint_steps.sort()
    
    selected_checkpoints = []
    for step, filename in checkpoint_steps:
        if step in selected_steps:
            selected_checkpoints.append((step, os.path.join(checkpoint_dir, filename)))
    
    # Patch the generate_handwriting.py file to allow using regular model
    if args.compare_models:
        patch_file_for_model_choice()
    
    # Generate samples for each checkpoint and text
    sample_paths = {}
    
    for text in texts:
        sample_paths[text] = {'ema': [], 'regular': []}
        text_dir = os.path.join(comparison_dir, text)
        os.makedirs(text_dir, exist_ok=True)
        
        print(f"Generating samples for text: '{text}'")
        for step, checkpoint_path in tqdm(selected_checkpoints):
            # Generate EMA model sample
            ema_dir = os.path.join(text_dir, f"step_{step}_ema")
            os.makedirs(ema_dir, exist_ok=True)
            ema_path = generate_sample(checkpoint_path, ema_dir, text, use_ema=True)
            if ema_path:
                sample_paths[text]['ema'].append((step, ema_path))
            
            # Generate regular model sample if requested
            if args.compare_models:
                regular_dir = os.path.join(text_dir, f"step_{step}_regular")
                os.makedirs(regular_dir, exist_ok=True)
                regular_path = generate_sample(checkpoint_path, regular_dir, text, use_ema=False)
                if regular_path:
                    sample_paths[text]['regular'].append((step, regular_path))
    
    # Create comparison plots for each text
    for text in texts:
        if not sample_paths[text]['ema']:
            print(f"No samples generated for text: '{text}'")
            continue
        
        if args.compare_models:
            # Create plot comparing EMA vs regular models
            create_model_comparison_plot(text, sample_paths[text], comparison_dir)
        else:
            # Create plot showing progression over training steps
            create_progression_plot(text, sample_paths[text]['ema'], comparison_dir)
    
    print("Comparison completed. Results saved to:", comparison_dir)

def patch_file_for_model_choice():
    """Patch the generate_handwriting.py file to support choosing between EMA and regular model."""
    generate_file = "examples/images/IAM/generate_handwriting.py"
    
    with open(generate_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "USE_REGULAR_MODEL" in content:
        return
    
    # Find the checkpoint loading code
    checkpoint_loading_code = """        if "ema_model" in checkpoint:
            model.load_state_dict(checkpoint["ema_model"])
            print(f"Loaded EMA model from {FLAGS.checkpoint}")
        else:
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded model from {FLAGS.checkpoint}")"""
    
    # Replacement code with environment variable check
    replacement_code = """        # Check environment variable to decide which model to use
        use_regular_model = os.environ.get("USE_REGULAR_MODEL", "0") == "1"
        
        if "ema_model" in checkpoint and not use_regular_model:
            model.load_state_dict(checkpoint["ema_model"])
            print(f"Loaded EMA model from {FLAGS.checkpoint}")
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded regular model from {FLAGS.checkpoint}")
        else:
            # Fallback
            print("Warning: Could not find appropriate model in checkpoint")
            for key in checkpoint:
                if isinstance(checkpoint[key], dict) and any('weight' in k for k in checkpoint[key]):
                    model.load_state_dict(checkpoint[key])
                    print(f"Loaded fallback model from {FLAGS.checkpoint}")
                    break"""
    
    # Replace the code
    modified_content = content.replace(checkpoint_loading_code, replacement_code)
    
    # Write the modified file
    with open(generate_file, 'w') as f:
        f.write(modified_content)
    
    print("Patched generate_handwriting.py to support model selection")

def create_progression_plot(text, samples, output_dir):
    """Create a plot showing sample progression over training steps."""
    n_samples = len(samples)
    if n_samples == 0:
        return
    
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 4, 3))
    fig.suptitle(f"Generation progress for '{text}'")
    
    for i, (step, path) in enumerate(samples):
        try:
            img = Image.open(path)
            ax = axes[i] if n_samples > 1 else axes
            ax.imshow(np.array(img), cmap='gray')
            ax.set_title(f"Step {step}")
            ax.axis('off')
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"progression_{text}.png"), dpi=150)
    print(f"Saved progression plot for '{text}' to {os.path.join(output_dir, f'progression_{text}.png')}")

def create_model_comparison_plot(text, samples_dict, output_dir):
    """Create a plot comparing EMA vs regular model."""
    ema_samples = samples_dict['ema']
    regular_samples = samples_dict['regular']
    
    if not ema_samples or not regular_samples:
        return
    
    # Find common steps between EMA and regular
    ema_steps = dict(ema_samples)
    regular_steps = dict(regular_samples)
    common_steps = sorted(set(ema_steps.keys()).intersection(set(regular_steps.keys())))
    
    if not common_steps:
        return
    
    # Create a row for each step, with EMA and regular side by side
    n_steps = len(common_steps)
    fig, axes = plt.subplots(n_steps, 2, figsize=(8, n_steps * 3))
    fig.suptitle(f"EMA vs Regular Model for '{text}'")
    
    for i, step in enumerate(common_steps):
        # EMA model
        try:
            ema_img = Image.open(ema_steps[step])
            ax = axes[i, 0] if n_steps > 1 else axes[0]
            ax.imshow(np.array(ema_img), cmap='gray')
            ax.set_title(f"Step {step} - EMA")
            ax.axis('off')
        except Exception as e:
            print(f"Error loading EMA image for step {step}: {e}")
        
        # Regular model
        try:
            regular_img = Image.open(regular_steps[step])
            ax = axes[i, 1] if n_steps > 1 else axes[1]
            ax.imshow(np.array(regular_img), cmap='gray')
            ax.set_title(f"Step {step} - Regular")
            ax.axis('off')
        except Exception as e:
            print(f"Error loading Regular image for step {step}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_models_{text}.png"), dpi=150)
    print(f"Saved model comparison for '{text}' to {os.path.join(output_dir, f'comparison_models_{text}.png')}")

if __name__ == "__main__":
    main() 