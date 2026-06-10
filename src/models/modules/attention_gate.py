import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        F_g: Channels in the gating signal (from the decoder)
        F_l: Channels in the skip connection (from the encoder)
        F_int: Intermediate channels (usually half of F_g)
        """
        super().__init__()
        
        # Transform the gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transform the skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Calculate the attention weights (0 to 1)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine the signals and apply non-linearity
        psi = self.relu(g1 + x1)
        
        # Squash to a mask between 0 and 1
        attention_weights = self.psi(psi)
        
        # Multiply the original skip connection by the attention mask
        return x * attention_weights