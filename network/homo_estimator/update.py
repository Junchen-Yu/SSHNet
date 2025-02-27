import torch
import torch.nn as nn

class GMA(nn.Module):
    def __init__(self, input_dim=164, hidden_dim=64, output_dim=2, downsample=4):
        super(GMA, self).__init__()
        
        self.in_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        self.same_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1), 
                                                   nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                                                   nn.ReLU()) for i in range(2)])
        
        self.down_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1), 
                                                       nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                                                       nn.ReLU(),
                                                       nn.MaxPool2d(kernel_size = 2, stride=2)) for i in range(downsample)])
        
        self.out_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, corr, flow):
        x = self.in_conv(torch.cat([corr, flow], dim=1))
        for layer in self.same_layers: x = layer(x)
        for layer in self.down_layers: x = layer(x)
        x = self.out_conv(x)
        return x

class CNN_64(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_64, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer10(x)
        return x


class CNN_32(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_32, self).__init__()
        
        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)
        return x


class GMA_update(nn.Module):
    def __init__(self, sz):
        super().__init__()
        if sz==32:
            self.cnn = CNN_32(80)
        if sz==64:
            self.cnn = CNN_64(64)
            
    def forward(self, corr_flow):      
        delta_flow = self.cnn(corr_flow)   
        return delta_flow