import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
from matplotlib import pyplot as plt

url = "/cluster/work/cotterell/mm_swissai/datasets/hdvila/1000_hd_vila_shuffled/0000000000.tar"
# url = "https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar"
ds = wds.WebDataset(url).decode(wds.torch_video) # .to_tuple("png", "json")
for sample in ds:
    break
# for image, json in pil_dataset:
#     break
# plt.imshow(image)
print()
# dataset = wds.WebDataset(url)

# # for sample in islice(dataset, 0, 3):
# #     for key, value in sample.items():
# #         print(key, repr(value)[:50])
# #     print()

# # dataset = (
# #     wds.WebDataset(url)
# #     .shuffle(100)
# #     .decode("rgb")
# #     .to_tuple("mp4", "json")
# # )

# # for image, data in islice(dataset, 0, 3):
# #     print(image.shape, image.dtype, type(data))

# batch_size = 20
# dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)
# images, targets = next(iter(dataloader))
# print(images.shape)