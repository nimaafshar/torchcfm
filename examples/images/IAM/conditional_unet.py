import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.models.unet.unet import UNetModel, timestep_embedding, normalization, zero_module, conv_nd, linear
import math

class CharacterEmbedding(nn.Module):
    """Embedding layer for character sequences."""
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        return self.embedding(x)

class CharacterConditionedUNet(UNetModel):
    """UNet model conditioned on character sequences by concatenating embeddings to the input channels.
    Inherits directly from UNetModel for clearer parameter passing.
    """
    
    def __init__(
        self,
        image_size,       # e.g., 128 (usually width)
        image_channels,   # Input image channels (e.g., 1)
        model_channels,   # Base channel count for the model (e.g., 128)
        num_res_blocks, 
        channel_mult,     # e.g., (1, 2, 2, 2)
        embed_dim,        # Conditioning embedding dim (e.g., 64)
        vocab_size,       # Size of character vocabulary
        attention_resolutions, # e.g., (16, 8)
        dropout=0,
        learn_sigma=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
        num_heads=1,             # Need these for base UNetModel
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_new_attention_order=False,
    ):
        self.cond_channels = embed_dim
        total_in_channels = image_channels + self.cond_channels
        out_channels = image_channels * 2 if learn_sigma else image_channels

        super().__init__(
            image_size=image_size,
            in_channels=total_in_channels, # Total channels entering the UNet
            model_channels=model_channels, # Base channel count
            out_channels=out_channels,     # Should output image channels (or *2 if sigma)
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=tuple(channel_mult), # Ensure it's a tuple
            conv_resample=True,
            dims=2,
            num_classes=None, # Not class-conditional
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
        
        # Character embedding layer
        self.char_embedding = CharacterEmbedding(vocab_size, embed_dim)

    def forward(self, t, x, char_indices=None, *args, **kwargs):
        """
        Forward pass with character sequence conditioning via channel concatenation.
        
        Args:
            t: Time step [B]
            x: Input image tensor [B, C_img, H, W]
            char_indices: Character indices tensor [B, L] where L is sequence length
        """
        B, C_img_actual, H, W = x.shape
        
        if C_img_actual != (self.in_channels - self.cond_channels):
             raise ValueError(
                 f"Input image channels ({C_img_actual}) do not match "
                 f"model's expected image channels ({self.in_channels - self.cond_channels})"
             )

        # Create conditioning tensor
        if char_indices is None:
            cond = torch.zeros(B, self.cond_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            char_emb = self.char_embedding(char_indices)  # [B, L, embed_dim]
            char_emb_pooled = char_emb.mean(dim=1)      # [B, embed_dim]
            cond = char_emb_pooled[:, :, None, None]    # [B, cond_channels, 1, 1]
            cond = cond.repeat(1, 1, H, W)              # [B, cond_channels, H, W]

        # Concatenate conditioning tensor to the image tensor
        conditioned_x = torch.cat([x, cond], dim=1) # [B, C_img + cond_channels, H, W]
        
        # Call the base UNetModel's forward pass directly
        # Base forward expects (t, x, y=None), where x is the full input
        return super().forward(t, conditioned_x, y=None) 