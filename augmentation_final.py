import torch
import numpy as np

def randomC4(image):
    return torch.rot90(image, np.random.choice([-1,0,1,2]), [1,2])
def C4(image,i):
    return torch.rot90(image, i, [2,3])