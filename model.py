import torch
import torch.nn as nn
from typing import Optional, Tuple


class FrameNorm1D(nn.Module):
    """
    Frame-wise instance normalization over the frequency (feature) axis.
    Useful for causal sequence models that process one frame at a time.

    Input:  (B, C, T, F)
    Output: same shape, normalized over F per (B, T)
    """

    def __init__(self, C: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.in1d = nn.InstanceNorm1d(C, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        y = x.permute(0, 2, 1, 3).reshape(B * T, C, F)
        y = self.in1d(y)
        y = y.view(B, T, C, F).permute(0, 2, 1, 3)
        return y


class TemporalCRN(nn.Module):
    """
    Convolutional Recurrent Network (CRN) for generic sequence-to-sequence modeling
    in the time–frequency domain.

    This architecture processes complex-valued sequences (e.g., STFTs) using
    convolutional feature extraction along frequency, recurrent temporal modeling
    with LSTMs, and deconvolutional decoding with skip connections.

    Args:
        input_channels (int): Number of input channels (2 for real+imaginary).
        base_channels (int): Number of channels for the first conv layer.
        hidden_size (int): LSTM hidden dimension.
        num_layers (int): Number of LSTM layers.

    Input:
        Xc (torch.Tensor): (B, F, T) complex-valued tensor.
        state (tuple): Optional (h, c) for LSTM streaming inference.

    Returns:
        Yc (torch.Tensor): (B, F, T) complex-valued output.
        state (tuple): Updated (h, c) hidden state.
    """

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 4,
        hidden_size: int = 192,
        num_layers: int = 2,
    ):
        super().__init__()

        # ----- Encoder -----
        self.conv2d_1 = nn.Conv2d(input_channels, base_channels, (1, 3), stride=(1, 2))
        self.norm_e1 = FrameNorm1D(base_channels)
        self.act_e1 = nn.PReLU(base_channels)

        self.conv2d_2 = nn.Conv2d(base_channels, 2 * base_channels, (1, 3), stride=(1, 2))
        self.norm_e2 = FrameNorm1D(2 * base_channels)
        self.act_e2 = nn.PReLU(2 * base_channels)

        self.conv2d_3 = nn.Conv2d(2 * base_channels, 4 * base_channels, (1, 3), stride=(1, 2))
        self.norm_e3 = FrameNorm1D(4 * base_channels)
        self.act_e3 = nn.PReLU(4 * base_channels)

        self.conv2d_4 = nn.Conv2d(4 * base_channels, 8 * base_channels, (1, 3), stride=(1, 2))
        self.norm_e4 = FrameNorm1D(8 * base_channels)
        self.act_e4 = nn.PReLU(8 * base_channels)

        self.conv2d_5 = nn.Conv2d(8 * base_channels, 16 * base_channels, (1, 3), stride=(1, 2))
        self.norm_e5 = FrameNorm1D(16 * base_channels)
        self.act_e5 = nn.PReLU(16 * base_channels)

        # ----- Temporal core -----
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # ----- Decoder -----
        self.deconv2d_5 = nn.ConvTranspose2d(128, 32, (1, 3), stride=(1, 2))
        self.norm_d5 = FrameNorm1D(32)
        self.act_d5 = nn.PReLU(32)

        self.deconv2d_4 = nn.ConvTranspose2d(64, 16, (1, 3), stride=(1, 2))
        self.norm_d4 = FrameNorm1D(16)
        self.act_d4 = nn.PReLU(16)

        self.deconv2d_3 = nn.ConvTranspose2d(32, 8, (1, 3), stride=(1, 2))
        self.norm_d3 = FrameNorm1D(8)
        self.act_d3 = nn.PReLU(8)

        self.deconv2d_2 = nn.ConvTranspose2d(16, 4, (1, 3), stride=(1, 2), output_padding=(0, 1))
        self.norm_d2 = FrameNorm1D(4)
        self.act_d2 = nn.PReLU(4)

        self.deconv2d_1 = nn.ConvTranspose2d(8, 2, (1, 3), stride=(1, 2))

    def forward(
        self, Xc: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        assert torch.is_complex(Xc), "Expected complex-valued input (B, F, T)."
        Xc_ref = Xc

        # Convert complex → real-valued representation
        x = torch.view_as_real(Xc).permute(0, 3, 2, 1)  # (B, 2, T, F)

        # ----- Encoder -----
        x1 = self.act_e1(self.norm_e1(self.conv2d_1(x)))
        x2 = self.act_e2(self.norm_e2(self.conv2d_2(x1)))
        x3 = self.act_e3(self.norm_e3(self.conv2d_3(x2)))
        x4 = self.act_e4(self.norm_e4(self.conv2d_4(x3)))
        x5 = self.act_e5(self.norm_e5(self.conv2d_5(x4)))

        # ----- LSTM -----
        z = x5.permute(0, 2, 1, 3).reshape(x5.size(0), x5.size(2), -1)
        z, state = self.lstm(z, state)
        z = z.reshape(x5.size(0), x5.size(2), 64, 3).permute(0, 2, 1, 3)

        # ----- Decoder -----
        y5 = self.act_d5(self.norm_d5(self.deconv2d_5(torch.cat([z, x5], dim=1))))
        y4 = self.act_d4(self.norm_d4(self.deconv2d_4(torch.cat([y5, x4], dim=1))))
        y3 = self.act_d3(self.norm_d3(self.deconv2d_3(torch.cat([y4, x3], dim=1))))
        y2 = self.act_d2(self.norm_d2(self.deconv2d_2(torch.cat([y3, x2], dim=1))))

        y1 = self.deconv2d_1(torch.cat([y2, x1], dim=1))
        m_ri = torch.tanh(y1).permute(0, 3, 2, 1).contiguous()
        M = torch.view_as_complex(m_ri)  # (B, F, T)

        Yc = M * Xc_ref  # Apply complex ratio mask
        return Yc, state
