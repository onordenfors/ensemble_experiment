import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# A simple CNN with average pooling and layer normalization
class CNN(nn.Module):
    def __init__(self,hidden=(16,16,16)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden[0], 3, bias=False,stride=1,padding=1)
        self.conv2 = nn.Conv2d(hidden[0], hidden[1], 3, bias=False,stride=1,padding=1)
        self.conv3 = nn.Conv2d(hidden[1], hidden[2], 3, bias=False,stride=1,padding=1)
        self.lin1 = nn.Linear(hidden[2]*7*7,10,bias=False)
        self.pool = nn.AvgPool2d(2)
        self.activate = nn.Tanh()
        self.ln1 = nn.LayerNorm(normalized_shape=(14,14),elementwise_affine=False)
        self.ln2 = nn.LayerNorm(normalized_shape=(7,7),elementwise_affine=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
    
    def forward(self,x):
        x=self.ln1(self.activate(self.pool(self.conv1(x))))
        x=self.ln2(self.activate(self.pool(self.conv2(x))))
        x=self.ln2(self.activate(self.conv3(x)))
        h = x.view(x.shape[0], -1)
        x = self.lin1(h)
        return x

class omegaConv(nn.Module):
    # convolution layer with (2k+1)x(2k+1) filters having a given symmetric support
    def __init__(self, in_channels: int, out_channels: int, omega, orthogonal=True):
        super().__init__()

        # Set padding depending on the size of the filter
        self.padding = int((omega.shape[0] - 1)/2)

        # Detect the pattern of the support
        pattern = omega.nonzero()
        self.pattern = pattern
        
        self.orthogonal=orthogonal

        # Create and register buffer for stacked support mask
        stacked_mask = torch.zeros(pattern.shape[0],omega.shape[0],omega.shape[1])

        for i in range(pattern.shape[0]):
            stacked_mask[i,pattern[i,:][0],pattern[i,:][1]] = 1
        # If orthogonal is set to false, generate a non-orthogonal basis instead.
        if not orthogonal:
            basis = torch.zeros(pattern.shape[0],omega.shape[0],omega.shape[1])
            coeff = torch.randn((pattern.shape[0],pattern.shape[0]))
            for i in range(pattern.shape[0]):
                for j in range(pattern.shape[0]):
                    basis[i,:,:] += coeff[i,j]*stacked_mask[j,:,:]
                basis[i,:,:] = basis[i,:,:] / torch.norm(basis[i,:,:])
            self.register_buffer('stacked_mask',basis)
        else:
            self.register_buffer('stacked_mask',stacked_mask)

        # Initialize weight tensor
        self.weight = nn.Parameter(torch.empty(out_channels,in_channels,pattern.shape[0]))
        torch.nn.init.normal_(self.weight)
    
    def load_weights(self,layer):
        # Function to load model weights
        weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            pattern = self.pattern
            coeffs = layer.calculate_weight_tensor()
            if self.orthogonal:
                for i in range(pattern.shape[0]):
                    weight[:,:,i] = coeffs[:,:,pattern[i,:][0],pattern[i,:][1]]
            else:
                L = self.stacked_mask.flatten(-2,-1).transpose(-2,-1)
                phi = coeffs.flatten(-2,-1)
                w = torch.einsum('ij,bli->blj',L,phi)
                weight = torch.linalg.solve(L.T@L,w.transpose(-2,-1))
                weight = torch.transpose(weight,-2,-1)
            self.weight = nn.Parameter(weight)

    def calculate_weight_tensor(self):
        # Calculates weight tensor with given support

        # Parameters and masks
        weight = self.weight
        mask = self.stacked_mask

        # Unsqueeze to match dimensions
        weight = torch.unsqueeze(torch.unsqueeze(weight,-1),-1)
        mask = torch.unsqueeze(torch.unsqueeze(mask,0),0)

        # Calculate weight
        weight = torch.sum(torch.mul(weight,mask),dim=2)

        return weight
    
    def projection(self):
        # Calculates projection onto the equivariant subspace
        weight = self.calculate_weight_tensor()
        proj = weight + torch.rot90(weight,1,[2,3]) + torch.rot90(weight,2,[2,3]) + torch.rot90(weight,3,[2,3])
        proj /= 4
        mask = self.stacked_mask
        mask = torch.unsqueeze(torch.unsqueeze(mask,0),0)
        mask = torch.sum(torch.abs(mask),dim=2)
        mask[mask!=0] = 1
        proj = proj*mask
        proj = proj*torch.rot90(mask,1,(-2,-1))
        proj = proj*torch.rot90(mask,2,(-2,-1))
        proj = proj*torch.rot90(mask,3,(-2,-1))
        return proj

    def forward(self,x):
        weight = self.calculate_weight_tensor()
        return F.conv2d(x,weight,bias=None,stride=1,padding=self.padding)

class omegaConvAsym(nn.Module):
    # convolution layer with (2k+1)x(2k+1) filters having a given asymmetric support
    def __init__(self, in_channels: int, out_channels: int, omega, orthogonal=True):
        super().__init__()

        # Set padding depending on the size of the filter
        self.padding = int((omega.shape[0] - 1)/2)

        # Detect the pattern of the support
        pattern = omega.nonzero()
        self.pattern = pattern
        
        self.orthogonal=orthogonal

        # Create and register buffer for stacked support mask
        stacked_mask = torch.zeros(pattern.shape[0],omega.shape[0],omega.shape[1])

        for i in range(pattern.shape[0]):
            stacked_mask[i,pattern[i,:][0],pattern[i,:][1]] = 1
        # If orthogonal is set to false, generate a non-orthogonal basis instead.
        if not orthogonal:
            basis = torch.zeros(pattern.shape[0],omega.shape[0],omega.shape[1])
            coeff = torch.randn((pattern.shape[0],pattern.shape[0]))
            for i in range(pattern.shape[0]):
                for j in range(pattern.shape[0]):
                    basis[i,:,:] += coeff[i,j]*stacked_mask[j,:,:]
                basis[i,:,:] = basis[i,:,:] / torch.norm(basis[i,:,:])
            self.register_buffer('stacked_mask',basis)
        else:
            self.register_buffer('stacked_mask',stacked_mask)

        # Initialize weight tensor
        self.weight = nn.Parameter(torch.empty(out_channels,in_channels,pattern.shape[0]))
        torch.nn.init.normal_(self.weight)
        # Zero out asymmetric parts of the filter
        for i in range(self.padding*(self.padding+1)):
            if i%self.padding == 0:
                pass
            else:
                torch.nn.init.constant_(self.weight[:,:,i],0)

    
    def load_weights(self,layer):
        # Function to load model weights
        weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            pattern = self.pattern
            coeffs = layer.calculate_weight_tensor()
            if self.orthogonal:
                for i in range(pattern.shape[0]):
                    weight[:,:,i] = coeffs[:,:,pattern[i,:][0],pattern[i,:][1]]
            else:
                L = self.stacked_mask.flatten(-2,-1).transpose(-2,-1)
                phi = coeffs.flatten(-2,-1)
                w = torch.einsum('ij,bli->blj',L,phi)
                weight = torch.linalg.solve(L.T@L,w.transpose(-2,-1))
                weight = torch.transpose(weight,-2,-1)
            self.weight = nn.Parameter(weight)

    def calculate_weight_tensor(self):
        # Calculates weight tensor with given support

        # Parameters and masks
        weight = self.weight
        mask = self.stacked_mask

        # Unsqueeze to match dimensions
        weight = torch.unsqueeze(torch.unsqueeze(weight,-1),-1)
        mask = torch.unsqueeze(torch.unsqueeze(mask,0),0)

        # Calculate weight
        weight = torch.sum(torch.mul(weight,mask),dim=2)

        return weight
    
    def projection(self):
        # Calculates projection onto the equivariant subspace
        weight = self.calculate_weight_tensor()
        proj = weight + torch.rot90(weight,1,[2,3]) + torch.rot90(weight,2,[2,3]) + torch.rot90(weight,3,[2,3])
        proj /= 4
        mask = self.stacked_mask
        mask = torch.unsqueeze(torch.unsqueeze(mask,0),0)
        mask = torch.sum(torch.abs(mask),dim=2)
        mask[mask!=0] = 1
        proj = proj*mask
        proj = proj*torch.rot90(mask,1,(-2,-1))
        proj = proj*torch.rot90(mask,2,(-2,-1))
        proj = proj*torch.rot90(mask,3,(-2,-1))
        return proj

    def forward(self,x):
        weight = self.calculate_weight_tensor()
        return F.conv2d(x,weight,bias=None,stride=1,padding=self.padding)
        
class omegaCNN(nn.Module):
    # Convolutional neural network, with a given support for the kernel
    def __init__(self,omega,lowest_im_dim=7,out_dim=10,hidden=(16,16,16),orthogonal=True):
        super().__init__()
        self.lowest_im_dim = lowest_im_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.conv1 = omegaConv(1,hidden[0],omega,orthogonal)
        self.conv2 = omegaConv(hidden[0],hidden[1],omega,orthogonal)
        self.conv3 = omegaConv(hidden[1],hidden[2],omega,orthogonal)
        self.lin = nn.Linear(hidden[2]*lowest_im_dim*lowest_im_dim,out_dim,bias=False)
        self.pool = nn.AvgPool2d(2)
        self.activate = nn.Tanh()
        self.bn1 = nn.LayerNorm(normalized_shape=(14,14),elementwise_affine=False)
        self.bn2 = nn.LayerNorm(normalized_shape=(7,7),elementwise_affine=False)

    def is_equivariant(self):
        return False
    
    def load_weights(self, model):
        # Calls the weight loading function in each convolution layer and
        # loads the weights from the linear layer.
        with torch.no_grad():
            weight = model.lin.calculate_weight_tensor()
            self.lin.weight = nn.Parameter(weight)
            self.conv1.load_weights(model.conv1)
            self.conv2.load_weights(model.conv2)
            self.conv3.load_weights(model.conv3)

    def projection(self):
        # Calls the projection function in each convolution layer and
        # calculates the projection in the linear layer.
        proj_conv1 = self.conv1.projection()
        proj_conv2 = self.conv2.projection()
        proj_conv3 = self.conv3.projection()
        weight = self.lin.weight
        proj = torch.reshape(weight, (self.out_dim,self.hidden[2],self.lowest_im_dim,self.lowest_im_dim))
        proj = proj+torch.rot90(proj,1,[2,3])+torch.rot90(proj,2,[2,3])+torch.rot90(proj,3,[2,3])
        proj /= 4
        proj_lin = torch.reshape(proj, (self.out_dim, self.hidden[2]*self.lowest_im_dim*self.lowest_im_dim))
        return proj_conv1, proj_conv2, proj_conv3, proj_lin
    
    def calculate_weight_tensor(self):
        # Calls the weight calculation function in each convolution layer
        # and calculates the weights in the linear layer.
        weight_conv1 = self.conv1.calculate_weight_tensor()
        weight_conv2 = self.conv2.calculate_weight_tensor()
        weight_conv3 = self.conv3.calculate_weight_tensor()
        weight_lin = self.lin.weight
        return weight_conv1, weight_conv2, weight_conv3, weight_lin
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight)


    def forward(self,x):
        x = self.activate(self.pool(self.conv1(x)))
        x = self.bn1(x)
        x = self.activate(self.pool(self.conv2(x)))
        x = self.bn2(x)
        x = self.activate(self.conv3(x))
        x = self.bn2(x)
        h = x.view(x.shape[0], -1)
        x = self.lin(h)
        return x
    
class omegaCNNAsym(nn.Module):
    # Convolutional neural network, with a given support for the kernel
    def __init__(self,omega,lowest_im_dim=7,out_dim=10,hidden=(16,16,16),orthogonal=True):
        super().__init__()
        self.lowest_im_dim = lowest_im_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.conv1 = omegaConvAsym(1,hidden[0],omega,orthogonal)
        self.conv2 = omegaConvAsym(hidden[0],hidden[1],omega,orthogonal)
        self.conv3 = omegaConvAsym(hidden[1],hidden[2],omega,orthogonal)
        self.lin = nn.Linear(hidden[2]*lowest_im_dim*lowest_im_dim,out_dim,bias=False)
        self.pool = nn.AvgPool2d(2)
        self.activate = nn.Tanh()
        self.bn1 = nn.LayerNorm(normalized_shape=(14,14),elementwise_affine=False)
        self.bn2 = nn.LayerNorm(normalized_shape=(7,7),elementwise_affine=False)

    def is_equivariant(self):
        return False
    
    def load_weights(self, model):
        # Calls the weight loading function in each convolution layer and
        # loads the weights from the linear layer.
        with torch.no_grad():
            weight = model.lin.calculate_weight_tensor()
            self.lin.weight = nn.Parameter(weight)
            self.conv1.load_weights(model.conv1)
            self.conv2.load_weights(model.conv2)
            self.conv3.load_weights(model.conv3)

    def projection(self):
        # Calls the projection function in each convolution layer and
        # calculates the projection in the linear layer.
        proj_conv1 = self.conv1.projection()
        proj_conv2 = self.conv2.projection()
        proj_conv3 = self.conv3.projection()
        weight = self.lin.weight
        proj = torch.reshape(weight, (self.out_dim,self.hidden[2],self.lowest_im_dim,self.lowest_im_dim))
        proj = proj+torch.rot90(proj,1,[2,3])+torch.rot90(proj,2,[2,3])+torch.rot90(proj,3,[2,3])
        proj /= 4
        proj_lin = torch.reshape(proj, (self.out_dim, self.hidden[2]*self.lowest_im_dim*self.lowest_im_dim))
        return proj_conv1, proj_conv2, proj_conv3, proj_lin
    
    def calculate_weight_tensor(self):
        # Calls the weight calculation function in each convolution layer
        # and calculates the weights in the linear layer.
        weight_conv1 = self.conv1.calculate_weight_tensor()
        weight_conv2 = self.conv2.calculate_weight_tensor()
        weight_conv3 = self.conv3.calculate_weight_tensor()
        weight_lin = self.lin.weight
        return weight_conv1, weight_conv2, weight_conv3, weight_lin
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight)


    def forward(self,x):
        x = self.activate(self.pool(self.conv1(x)))
        x = self.bn1(x)
        x = self.activate(self.pool(self.conv2(x)))
        x = self.bn2(x)
        x = self.activate(self.conv3(x))
        x = self.bn2(x)
        h = x.view(x.shape[0], -1)
        x = self.lin(h)
        return x

class Ensemble(nn.Module):
    def __init__(self,members):
        super().__init__()
        self.members = members

    def forward(self,x):
        sum=0
        n = 0
        for i, member in enumerate(self.members):
            sum = sum + member(x)
            n = n + 1
        x = sum/n
        return x