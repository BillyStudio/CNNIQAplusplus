from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from main import CNNIQAplusplusnet
from IQADataset import LocalNormalization, OverlappingCropPatches

x = Image.open('womanhat.bmp').convert('L')
#x = OverlappingCropPatches(x)
#x = torch.stack(x)
print(type(x))
trans = transforms.ToTensor()
x = trans(x)

x = x.unsqueeze(0)
x = x.unsqueeze(0)
#x = x.permute(3, 0, 1, 2)
print(x.size(-4), x.size(-3), x.size(-2), x.size(-1))

params = torch.load('./checkpoints/CNNIQAplusplus-LIVE-EXP0-lr=0.0001')
model = CNNIQAplusplusnet(n_distortions=5,ker_size=3,n1_kers=8,pool_size=2,n2_kers=50,n1_nodes=128,n2_nodes=512)
model.load_state_dict(params)
print(model)

q, d = model(x)
print(q.data.numpy())
print('score =', torch.mean(q))
