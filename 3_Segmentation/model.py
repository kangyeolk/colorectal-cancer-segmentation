import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BinaryClassifier, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024*16, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes))
        self._initialize_weights()
    
    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


'''
Append padding to keep concat size & input-output size
'''
class Conv3x3(nn.Module):
    """ Double Convolution 3x3 """
    def __init__(self, in_dim, out_dim):
        super(Conv3x3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim))

    def forward(self, x):
        out = self.block(x)
        return out

class UNet(nn.Module):
    """ Depth 2 UNet """
    def __init__(self, num_classes, in_dim=3, conv_dim=64):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.conv_dim = conv_dim
        self.build_unet()

    def build_unet(self):
        self.enc1 = Conv3x3(in_dim=3, out_dim=self.conv_dim)
        self.enc2 = Conv3x3(in_dim=self.conv_dim, out_dim=2*self.conv_dim)
        
        self.floor = Conv3x3(in_dim=2*self.conv_dim, out_dim=4*self.conv_dim)

        self.dec1 = Conv3x3(in_dim=4*self.conv_dim, out_dim=2*self.conv_dim)
        self.dec2 = Conv3x3(in_dim=2*self.conv_dim, out_dim=1*self.conv_dim)

        self.last = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.num_classes, kernel_size=1),
            nn.Softmax(dim=1))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(4*self.conv_dim, 2*self.conv_dim, 
                                      kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(2*self.conv_dim, self.conv_dim, 
                                      kernel_size=2, stride=2)

    def forward(self, input):

        # Encoding
        h1 = self.enc1(input)
        h1_pool = self.maxpool(h1)
        h2 = self.enc2(h1_pool)
        h2_pool = self.maxpool(h2)

        # Floor
        h3 = self.floor(h2_pool)

        # Decoding
        h3_up = self.up1(h3)
        h4 = self.dec1(torch.cat([h2, h3_up], dim=1))
        h4_up = self.up2(h4)
        h5 = self.dec2(torch.cat([h1, h4_up], dim=1))

        # Last
        out = self.last(h5)

        assert input.size(-1) == out.size(-1) \
               ,'input size(W)-{} mismatches with output size(W)-{}' \
                .format(input.size(-1), out.size(-1))
        assert input.size(-2) == out.size(-2) \
               , 'input size(H)-{} mismatches with output size(H)-{}' \
                .format(input.size(-1), out.size(-1))
        
        return out

if __name__ == '__main__':
    sample = torch.randn((2, 3, 128, 128))
    model = UNet(num_classes=2)
    #print(model(sample).size())
    print(model(sample))
