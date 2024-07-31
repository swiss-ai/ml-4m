import cv2
import numpy as np
import torch

from fourm.utils import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, denormalize
from fourm.vq.vqvae import DiVAE

video = np.load("/store/swissai/a08/data/4m/video_rgb_tok_full/extract/00000.npy")
print(video.shape)

tokens = torch.from_numpy(video).long().cuda()
# shape: n_frames x n_channels x 196 (?) (e.g., 1341 x 3 x 196)

tok = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").cuda()

IMAGE_SIZE = 224

rgb_b3hw = tok.decode_tokens(tokens, image_size=IMAGE_SIZE, timesteps=50)
rgb_b3hw = denormalize(
    rgb_b3hw, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
)
# this is a video
for i, frame in enumerate(rgb_b3hw):
    frame = frame.permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).astype(np.uint8)
    cv2.imwrite(
        f"/store/swissai/a08/data/4m/video_rgb_tok_full/extract/{i:05d}.png", frame
    )
