# Data processing
'''Completely optional, this code can be use to add synthetic backgrounds'''
import torch
import random
import numpy as np
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    convert_to_images,
    one_hot_from_int,
)
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
# boilerplate pytorch code enforcing reproducibility
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

'''
In this cell, we create a dataset of synthetic images generated using BigGAN. We will create a total of 1000 batches of synthetic images, meaning that
there are a total of 3000 training images.

'''

# load the 256x256 model
import os
from tqdm import trange

if not os.path.exists('./synthetic_dataset.npy'):
    model = BigGAN.from_pretrained('biggan-deep-256').to(device).eval()

    num_passes = 3000
    batch_size = 1

    truncation = 0.4

    background_images = []

    for _ in trange(num_passes):
        class_vector = torch.from_numpy(
            one_hot_from_int(np.random.randint(0, 1000, size=batch_size), batch_size=batch_size)
        ).to(device)
        noise_vector = torch.from_numpy(
            truncated_noise_sample(truncation=truncation, batch_size=batch_size)
        ).to(device)
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation).cpu()
            background_images.extend(convert_to_images(output))
    np.save('./synthetic_dataset.npy',background_images,  allow_pickle=True)
    del model
    torch.cuda.empty_cache()
