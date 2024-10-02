import numpy as np
import torch
from tqdm import tqdm

def train(model, loader, optimizer, criterion, device):
    model.train()
    for data in tqdm(loader,desc='Batches completed:',leave=False):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
    return 