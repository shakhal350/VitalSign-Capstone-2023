import torch
from torch import nn
from transformers import ViTModel, ViTConfig
import math
import torch.nn.functional as F



class MAEModel(nn.Module):
    def __init__(self, input_dim=128, seq_len=128, mask_ratio=0.4):
        super(MAEModel, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        # Phase values are single-dimensional
        self.embedding = nn.Linear(1, input_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=16, dim_feedforward=2048)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.encoder_positional = nn.Parameter(
            torch.zeros(1, seq_len, input_dim))

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=16, dim_feedforward=2048)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=5)
        # Keep the same dimension just for simplicity
        self.decoder_positional = nn.Parameter(
            torch.zeros(1, seq_len, input_dim))
        self.decoder_linear = nn.Linear(input_dim, 1)  # Predicting a single value

        self.layer_norm = nn.LayerNorm(input_dim)


    def forward(self, x):
        if x.dim() == 2:  
            x = x.unsqueeze(-1)  
        x = self.embedding(x)  

        seq_len = x.size(1)  
        if self.encoder_positional.size(1) < seq_len:
            needed_size = seq_len - self.encoder_positional.size(1)
            extended_pos = self.encoder_positional[:, :needed_size, :].repeat(1, math.ceil(seq_len / self.encoder_positional.size(1)), 1)
            self.encoder_positional = torch.cat([self.encoder_positional, extended_pos[:, :needed_size, :]], dim=1)
        current_encoder_positional = self.encoder_positional[:, :seq_len, :]

        x = x.permute(1, 0, 2)  # Adjust for transformer which expects [seq_len, batch_size, feature_dim]
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)
        # applying a final linear layer to predict heart rate:
        output = self.decoder_linear(decoded.permute(1, 0, 2))

        return output


class ViTForHeartRateEstimation(nn.Module):
    def __init__(self, mae_encoder, seq_len=128, embed_dim=768):
        super(ViTForHeartRateEstimation, self).__init__()
        # Store the MAE encoder
        self.mae_encoder = mae_encoder

        # Vision Transformer Configuration
        self.config = ViTConfig(
            image_size=128,  # square shape output from MAE encoder
            patch_size=16,  # size of the patches to be extracted from the "image"
            num_channels=1,  # num of image channels
            num_attention_heads=16,  # num of attention heads in the ViT
            num_hidden_layers=5,  
            hidden_size=embed_dim,  
            mlp_dim=2048  # Dimensionality of the MLP (FeedForward) layer
        )
        self.vit = ViTModel(self.config)

        # Linear layer to predict the heart rate from the ViT token embedding
        self.head = nn.Linear(self.config.hidden_size, 1)
   

    def forward(self, inputs):
        encoded_features = self.mae_encoder(inputs)
        batch_size, seq_len, feature_dim = encoded_features.shape

        total_features = seq_len * feature_dim
        next_square = math.ceil(math.sqrt(total_features)) ** 2  # Find the next perfect square
        padding_needed = next_square - total_features

        if padding_needed > 0:
            # Add padding to make total features a perfect square
            padded_features = F.pad(encoded_features.reshape(batch_size, -1), (0, padding_needed), "constant", 0)
            pseudo_image_side = int(math.sqrt(next_square))
            encoded_images = padded_features.view(batch_size, 1, pseudo_image_side, pseudo_image_side)
        else:
            pseudo_image_side = int(math.sqrt(total_features))
            encoded_images = encoded_features.view(batch_size, 1, pseudo_image_side, pseudo_image_side)

        if pseudo_image_side != self.config.image_size:
            encoded_images = F.interpolate(encoded_images, size=(self.config.image_size, self.config.image_size), mode='bilinear', align_corners=False)

        outputs = self.vit(pixel_values=encoded_images).last_hidden_state
        heart_rate_estimate = self.head(outputs[:, 0])

        return heart_rate_estimate



def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
