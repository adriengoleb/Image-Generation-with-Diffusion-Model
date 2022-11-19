import os
from tqdm import trange
import torch
import torchvision
from utils import sample_timestep, load_model
from model import SimpleUnet
from params import img_size, device, T, model_name


if __name__ == '__main__':

    print('Model Loading...')
    model = SimpleUnet()#.cuda()
    model = load_model(model)
    model.eval()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print('Model loaded.')

    print('Start Generating :')
    os.makedirs('samples', exist_ok=True)
    with trange(1024, desc="Generated", unit="img") as te:
        for idx in te:
            img = torch.randn((1, 1, img_size, img_size), device=device)
            for i in range(0,T)[::-1]: # from 699 to 0
                t = torch.full((1,), i, device=device, dtype=torch.long)
                img = sample_timestep(model, img, t)
            img = torch.clamp(img, 0, 1) # NEW
            torchvision.utils.save_image(img, os.path.join('samples', f'{idx}.png'))