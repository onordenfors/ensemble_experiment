import torch
import torch.nn.functional as F
from augmentation_final import C4
from tqdm import tqdm, trange

def test(model, loader, criterion, device, G):

    running_loss = 0.0
    running_acc = 0.0
    osp = torch.empty((0))
    osp = osp.to(device)
    div = 0
    div_max = div
    
    model.eval()
    with torch.no_grad():

        for data in tqdm(loader,desc='Batches completed:',leave=False):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs,labels)

            one_hot_labels = F.one_hot(labels,10)
            correct = 0
            correct += (outputs.argmax(1) == one_hot_labels.argmax(1)).sum().item()
            correct = correct / labels.shape[0]
            
            probs = F.softmax(outputs,dim=1)
            log_probs = F.log_softmax(outputs,dim=1)
            part_osp = torch.zeros((inputs.shape[0]))
            part_osp = part_osp.to(device)
            for g in range(G):
                orbit_outputs = model(C4(inputs,g))
                part_osp += (orbit_outputs.argmax(1) == outputs.argmax(1))
                orbit_probs = F.softmax(orbit_outputs,dim=1)
                orbit_log_probs = F.log_softmax(orbit_outputs,dim=1)
                div_current = F.kl_div(orbit_log_probs, probs,reduction='batchmean').item() + F.kl_div(log_probs, orbit_probs,reduction='batchmean').item()
                if div_current > div_max:
                    div_max = div_current
                div += div_current
            osp = torch.cat((osp,part_osp), 0)

            running_loss += loss.item()
            running_acc += correct
    return running_loss/len(loader), running_acc/len(loader), torch.mean(osp).item(), torch.std(osp).item(), div/(G*len(loader)), div_max