import math

import torch
from torch import nn
import torch.nn.functional as F


def time_embedding(t, dim=32):
    k = torch.arange(dim // 2, device=t.device)
    angles = t[:, None] * torch.exp(-math.log(10000) * 2 * k / dim)
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class NoiseBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm = nn.GroupNorm(8, outputs)
        self.time = nn.Linear(64, outputs)
        self.out = nn.Sequential(nn.Conv2d(outputs, outputs, 3, padding=1),
                                 nn.GroupNorm(8, outputs), nn.SiLU())

    def forward(self, x, t):
        h = self.norm(self.conv(x)) + self.time(t)[:, :, None, None]
        return self.out(F.silu(h))


class DiffusionModel(nn.Module):
    def __init__(self, steps=100):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))
        self.enc = NoiseBlock(3, 32)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.mid = NoiseBlock(64, 64)
        self.down2 = nn.Conv2d(64, 96, 4, 2, 1)
        self.bottom = NoiseBlock(96, 96)
        self.up2 = nn.ConvTranspose2d(96, 64, 4, 2, 1)
        self.dec2 = NoiseBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec1 = NoiseBlock(64, 32)
        self.out = nn.Conv2d(32, 3, 3, padding=1)

        s = torch.arange(steps + 1, dtype=torch.float64) / steps
        curve = torch.cos((s + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - curve[1:] / curve[:-1]).clamp(max=0.999).float()
        self.register_buffer('beta', beta)
        self.register_buffer('abar', (1 - beta).cumprod(0))

    def forward(self, x, t):
        t = self.time(time_embedding(t))
        a = self.enc(x, t)
        b = self.mid(self.down1(a), t)
        c = self.bottom(self.down2(b), t)
        b = self.dec2(torch.cat((self.up2(c), b), dim=1), t)
        a = self.dec1(torch.cat((self.up1(b), a), dim=1), t)
        return self.out(a)

    def loss(self, x):
        t = torch.randint(len(self.beta), (len(x),), device=x.device)
        ab = self.abar[t, None, None, None]
        noise = torch.randn_like(x)
        noisy = ab.sqrt() * x + (1 - ab).sqrt() * noise
        return F.mse_loss(self(noisy, t), noise)

    @torch.no_grad()
    def sample(self, count):
        x = torch.randn(count, 3, 32, 32, device=self.beta.device)
        for i in reversed(range(len(self.beta))):
            t = torch.full((count,), i, device=x.device, dtype=torch.long)
            ab = self.abar[i]
            prev = self.abar[i - 1] if i else torch.ones_like(ab)
            b = self.beta[i]
            clean = ((x - (1 - ab).sqrt() * self(x, t)) / ab.sqrt()).clamp(-1, 1)
            x = (prev.sqrt() * b * clean + (1 - b).sqrt() * (1 - prev) * x) / (1 - ab)
            if i:
                x += (b * (1 - prev) / (1 - ab)).sqrt() * torch.randn_like(x)
        return x


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 1))

    def forward(self, x):
        return self.net(x).flatten()

    def langevin(self, x, steps):
        for _ in range(steps):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                grad, = torch.autograd.grad(self(x).sum(), x)
            with torch.no_grad():
                x = (x - 2 * grad + 0.01 * torch.randn_like(x)).clamp(-1, 1)
        return x.detach()

    def sample(self, count):
        device = next(self.parameters()).device
        x = torch.empty(count, 3, 32, 32, device=device).uniform_(-1, 1)
        return self.langevin(x, 60)
