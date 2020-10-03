import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ConvInput(nn.Module):
    '''Convolution Layers for Visual Inputs'''
    def __init__(self):
        super(ConvInput, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        
        return x

class BaseLayer(nn.Module):
    '''Base Layer'''
    def __init__(self):
        super(BaseLayer, self).__init__()

    def train_(self, img, ques, ans):
        '''Train on a batch of train data'''
        self.optimizer.zero_grad()
        
        # forward pass
        output = self(img, ques)
        
        # loss and optimization
        loss = F.nll_loss(output, ans)
        loss.backward()
        self.optimizer.step()
        
        # accuracy
        pred = output.argmax(1)
        accuracy = pred.eq(ans.data).cpu().sum() * 100. / len(ans)
        
        return accuracy, loss
        
    def evaluate(self, img, ques, ans):
        '''Evaluate on a batch of test data'''
        # forward pass
        output = self(img, ques)
        # loss
        loss = F.nll_loss(output, ans)
        # accuracy
        pred = output.argmax(1)
        accuracy = pred.eq(ans.data).cpu().sum() * 100. / len(ans)
        
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'models/epoch_{:02d}.pth'.format(epoch))
    
    
class RNLayer(nn.Module):
    '''Relation Network Layer (g_θ)'''
    def __init__(self):
        super(RNLayer, self).__init__()
        
        self.g_fc1 = nn.Linear(62, 256) # 62 i.e. (24+2)*2+10
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)
        
    def forward(self, x, ques):
        '''Forward Pass through g_θ'''
        # x -> (b, 24, 5, 5)
        # que -> (b, 10)
        b, c, d, _ = x.size()
        
        # add coordinates to x (feature maps)
        self.build_coord_tensor(b, d) # coord_tensor -> (b, 2, 5, 5)
        if x.device.type == 'cuda':
            self.coord_tensor = self.coord_tensor.cuda()
        x_tagged = torch.cat([x, self.coord_tensor], 1) # (b, 24+2, 5, 5)
        x_flat = x_tagged.view(b, c+2, d*d).permute(0, 2, 1) # (b, 25, 24+2)
                
        # repeat question vector for casting everywhere
        ques = ques.unsqueeze(1) # (b, 1, 10)
        ques = ques.repeat(1, 25, 1) # (b, 25, 10)
        ques = ques.unsqueeze(2) # (b, 25, 1, 10)
        
        # create i-j pairs of objects
        x_i = x_flat.unsqueeze(1) # (b, 1, 25, 26)
        x_i = x_i.repeat(1, 25, 1, 1) # (b, 25, 25, 26)
        x_j = x_flat.unsqueeze(2)  # (b, 25, 1, 26)
        # add question vector
        x_j = torch.cat([x_j, ques], 3) # (b, 25, 1, 26+10)
        x_j = x_j.repeat(1, 1, 25, 1) # (b, 25, 25, 36)
        
        # cast the pairs together
        x_full = torch.cat([x_i, x_j], 3) # (b, 25, 25, 26*2+10)
        
        # flatten for passing through g_θ
        x_g = x_full.view(-1, 62) # (b * 25 * 25, 62)
        
        # forward pass through g_θ linear layers
        x_g = F.relu(self.g_fc1(x_g))
        x_g = F.relu(self.g_fc2(x_g))
        x_g = F.relu(self.g_fc3(x_g))
        x_g = F.relu(self.g_fc4(x_g))
        
        # reshape
        x_g = x_g.view(b, (d * d) * (d * d), 256) # (b, 625, 256)
        # sum
        x_g = x_g.sum(1).squeeze(1) # (b, 256)
        
        return x_g
        
    def build_coord_tensor(self, b, d):
        '''Returns the coordinates of the objects (d = 5)'''
        coords = torch.linspace(-d/2., d/2., d) # (5)
        x = coords.unsqueeze(0).repeat(d, 1) # (5, 5)
        y = coords.unsqueeze(1).repeat(1, d) # (5, 5)
        ct = torch.stack((x,y)) # (2, 5, 5)
        self.coord_tensor = ct.unsqueeze(0).repeat(b, 1, 1, 1) # (b, 2, 5, 5)
        
class RNModel(BaseLayer):
    '''Relation Network Model (CNN + g_θ + f_ϕ)'''
    def __init__(self, args):
        super(RNModel, self).__init__()
        
        l_rate = args.lr if args != None else 0.0001
        
        self.conv_input = ConvInput() # CNN features
        self.rel_layer = RNLayer() # g_θ
        # f_ϕ
        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_out = nn.Linear(256, 10) # outputs logits over answer vocabulary
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=l_rate)
        
    
    def forward(self, img, ques):
        '''Forward pass through the full Relation Network Augmented model'''
        x = self.conv_input(img) # (b, 24, 5, 5)
        x_g = self.rel_layer(x, ques)
        x_f = F.relu(self.f_fc1(x_g))
        x_f = F.relu(self.f_fc2(x_f))
        x_f = F.dropout(x_f)
        out = F.log_softmax(self.f_out(x_f), dim=1)
        
        return out