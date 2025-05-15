import numpy as np
import torch
from tqdm import tqdm

def train(model, loader, optimizer, criterion, device):
    model.train()
    for data in tqdm(loader):
        data = data.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,data.y)
        loss.backward()
        optimizer.step()
    return 