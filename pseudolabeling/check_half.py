import cv2
import numpy as np
import torch
import pdb
from einops import rearrange

from fourm.utils import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, denormalize
from fourm.vq.vqvae import DiVAE

# video = np.load("/store/swissai/a08/data/4m/video_rgb_tok_full/extract/00000.npy")
video = np.load("/store/swissai/a08/markusf/ml-4m/frame_44.npy")
# pdb.set_trace()
print(video.shape)

tokens = torch.from_numpy(video).long().cuda()
# shape: n_frames x n_channels x 196 (?) (e.g., 1341 x 3 x 196)

# reshape to B x 3 x 14 x 14
tokens = tokens.view(16, 14, 14)

tok = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").cuda()

IMAGE_SIZE = 224

rgb_b3hw = tok.decode_tokens(tokens[[0], :, :], image_size=IMAGE_SIZE, timesteps=50)
rgb_b3hw = denormalize(
    rgb_b3hw, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
)
print(rgb_b3hw.shape)
# this is a video
for i, frame in enumerate(rgb_b3hw):
    frame = frame.cpu().numpy()
    frame = (frame * 255).astype(np.uint8)
    cv2.imwrite(
        f"/store/swissai/a08/markusf/ml-4m/frame_44.png", frame
    )
