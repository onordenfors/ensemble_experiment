import torch
import numpy as np
from torchvision.transforms.functional import rotate, InterpolationMode
import numpy as np

def randomC4(image):
    return torch.rot90(image, np.random.choice([-1,0,1,2]), [1,2])
def C4(image,i):
    return torch.rot90(image, i, [2,3])

def randomC16(image, interp = InterpolationMode.BILINEAR):
    theta = np.random.choice(16)/16*360
    return rotate(image,angle = theta, interpolation=interp)

def C16(image,i, interp = InterpolationMode.BILINEAR):
    theta = i/16*360
    return rotate(image,angle = theta, interpolation=interp)