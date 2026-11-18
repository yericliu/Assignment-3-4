import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from advanced_models import DiffusionModel, EnergyModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def train(name, loader, device, epochs):
    model = (DiffusionModel() if name == 'diffusion' else EnergyModel()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    if name == 'energy':
        replay = torch.empty(2048, 3, 32, 32, device=device).uniform_(-1, 1)
    for epoch in range(epochs):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == 'diffusion':
                loss = model.loss(x)
            else:
                idx = torch.randperm(len(replay), device=device)[:len(x)]
                fresh = torch.rand(len(x), 1, 1, 1, device=device) < 0.05
                start = torch.where(fresh, torch.empty_like(x).uniform_(-1, 1), replay[idx])
                negative = model.langevin(start, 12)
                replay[idx] = negative
                real, fake = model(x), model(negative)
                loss = (real - fake).mean() + 0.1 * (real.square() + fake.square()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        print(f'{name} epoch {epoch+1}/{epochs}, loss {total/len(loader.dataset):.4f}', flush=True)
        path = OUT / f'{name}.pth'
        temporary = path.with_suffix('.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['diffusion', 'energy', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on', device, flush=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    data = datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=128, shuffle=True, num_workers=2)
    OUT.mkdir(exist_ok=True)
    for name in (['diffusion', 'energy'] if args.model == 'both' else [args.model]):
        train(name, loader, device, args.epochs)
