"""ConvLSTM cell + future frame predictor (compatibil input stacked ca v4_future)."""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(
        self, batch: int, spatial: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = spatial
        h0 = torch.zeros(batch, self.hidden_channels, h, w, device=device)
        c0 = torch.zeros(batch, self.hidden_channels, h, w, device=device)
        return h0, c0


class ConvLSTMStack(nn.Module):
    """Stack de straturi ConvLSTM; fiecare strat primește output hidden anterior."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        cells = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(ch_in, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

    def init_states(
        self, batch: int, spatial: tuple[int, int], device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            cell.init_state(batch, spatial, device) for cell in self.cells
        ]

    def forward(
        self,
        x: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """x: (B, C, H, W) — un pas temporal."""
        cur = x
        new_states = []
        for cell, state in zip(self.cells, states):
            h, c = cell(cur, state)
            new_states.append((h, c))
            cur = h
        return cur, new_states


class FutureFrameConvLSTM(nn.Module):
    """
    Procesează secvența t-k..t-1 pas cu pas prin ConvLSTM, apoi decodează cadru t.
    Acceptă input stacked (B, 3*k, H, W) — la fel ca U-Net din v4_future.
    """

    def __init__(
        self,
        n_input_frames: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 1,
        encoder_channels: int = 32,
    ):
        super().__init__()
        self.n_input_frames = n_input_frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, encoder_channels, 3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTMStack(
            in_channels=encoder_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 3, 1),
        )

    def _split_input(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = self.n_input_frames
        if c != 3 * t:
            raise ValueError(f"Aștept {3 * t} canale, am primit {c}")
        return x.view(b, t, 3, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._split_input(x)
        b, t, _, h, w = seq.shape
        device = x.device
        states = self.convlstm.init_states(b, (h, w), device)

        for i in range(t):
            frame = self.frame_encoder(seq[:, i])
            hidden, states = self.convlstm(frame, states)

        return torch.sigmoid(self.decoder(hidden))

    @staticmethod
    def anomaly_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean(dim=(1, 2, 3))
