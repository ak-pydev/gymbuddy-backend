import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSkeletonModel(nn.Module):
    def __init__(self, num_classes=120, num_joints=25, in_channels=3, hidden_dim=64):
        super().__init__()
        
        self.input_dim = num_joints * in_channels # 75
        
        # Simple TCN-like structure
        self.conv1 = nn.Conv1d(self.input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        
    def forward(self, x, mask=None):
        """
        x: (B, T, J, C)
        """
        B, T, J, C = x.shape
        
        # Flatten J and C -> (B, T, J*C)
        x = x.view(B, T, -1)
        
        # Permute for Conv1d: (B, J*C, T)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x) # (B, Channels, 1)
        x = x.flatten(1)
        
        out = self.fc(x)
        return out
