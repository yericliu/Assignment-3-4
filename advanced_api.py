from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from torchvision.transforms.functional import to_pil_image

from advanced_models import DiffusionModel, EnergyModel

router = APIRouter()


@lru_cache(maxsize=2)
def load_model(name):
    path = Path(__file__).resolve().parent / 'outputs' / f'{name}.pth'
    if not path.exists():
        raise HTTPException(503, f'Run python train_advanced.py --model {name} first.')
    model = DiffusionModel() if name == 'diffusion' else EnergyModel()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.eval().requires_grad_(False).to(device)


def generate(name):
    image = load_model(name).sample(1)[0].detach().cpu()
    buffer = BytesIO()
    to_pil_image(((image + 1) / 2).clamp(0, 1)).save(buffer, format='PNG')
    return Response(buffer.getvalue(), media_type='image/png')


@router.get('/generate-diffusion')
def generate_diffusion():
    return generate('diffusion')


@router.get('/generate-energy')
def generate_energy():
    return generate('energy')
