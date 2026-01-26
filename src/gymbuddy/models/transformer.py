import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SkeletonTransformer(nn.Module):
    def __init__(self, num_classes=120, num_joints=25, in_channels=3, 
                 d_model=256, nhead=4, num_layers=4, dropout=0.5, dim_feedforward=1024):
        super().__init__()
        
        self.input_dim = num_joints * in_channels
        
        # 1. Embedding
        self.embedding = nn.Linear(self.input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        """
        x: (B, T, J, C)
        """
        B, T, J, C = x.shape
        x = x.view(B, T, -1) # (B, T, J*C)
        
        x = self.embedding(x) # (B, T, d_model)
        x = self.pos_encoder(x)
        
        # Transformer
        # mask arg in forward could be used for padding mask if we had variable lengths.
        # But we resampled to fixed T=60. So no key padding mask needed usually.
        x = self.transformer_encoder(x)
        
        # Global Pooling (Average over T)
        x = x.mean(dim=1) # (B, d_model)
        
        x = self.norm(x)
        out = self.fc(x)
        
        return out
