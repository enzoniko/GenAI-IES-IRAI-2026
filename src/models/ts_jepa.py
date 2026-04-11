import torch
import torch.nn as nn
import src.configs as cfg
import copy
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class Tokenizer(nn.Module):
    def __init__(self, in_channels=4, patch_size=50, d_model=128):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.conv(x)
        return x.transpose(1, 2)

class TSJEPA(nn.Module):
    def __init__(self, in_channels=4, patch_size=None, d_model=None, nhead=None, num_layers=None, ema_decay=None):
        super().__init__()
        
        # Pull from centralized config if not explicitly provided
        patch_size = patch_size or cfg.JEPA_CONFIG['patch_size']
        d_model = d_model or cfg.JEPA_CONFIG['d_model']
        nhead = nhead or cfg.JEPA_CONFIG['nhead']
        num_layers = num_layers or cfg.JEPA_CONFIG['num_layers']
        ema_decay = ema_decay or cfg.JEPA_CONFIG['ema_decay']
        
        self.tokenizer = Tokenizer(in_channels, patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        predictor_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=2)
        
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.ema_decay = ema_decay
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def update_ema(self):
        with torch.no_grad():
            for context_param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                target_param.data = self.ema_decay * target_param.data + (1.0 - self.ema_decay) * context_param.data

    def get_z_macro(self, x):
        tokens = self.tokenizer(x)
        tokens = self.pos_enc(tokens)
        encoded = self.context_encoder(tokens)
        z_macro = encoded.mean(dim=1)
        return z_macro

    def forward(self, x):
        tokens = self.tokenizer(x)
        B, N, D = tokens.shape
        
        num_mask = N // 2
        rand_indices = torch.randperm(N, device=x.device)
        mask_indices = rand_indices[:num_mask]
        keep_indices = rand_indices[num_mask:]
        
        with torch.no_grad():
            target_tokens = tokens + self.pos_enc.pe[:N, :].unsqueeze(0)
            target_encoded = self.target_encoder(target_tokens)
            target_unobserved = target_encoded[:, mask_indices, :]
            
        observed_tokens = tokens[:, keep_indices, :]
        observed_pos = self.pos_enc.pe[keep_indices, :].unsqueeze(0).expand(B, -1, -1)
        observed_tokens = observed_tokens + observed_pos
        context_encoded = self.context_encoder(observed_tokens)
        
        pred_input = self.mask_token.expand(B, num_mask, -1)
        pred_pos = self.pos_enc.pe[mask_indices, :].unsqueeze(0).expand(B, -1, -1)
        pred_input = pred_input + pred_pos
        
        full_pred_input = torch.cat([context_encoded, pred_input], dim=1)
        predicted = self.predictor(full_pred_input)
        
        predicted_unobserved = predicted[:, -num_mask:, :]
        
        return predicted_unobserved, target_unobserved
