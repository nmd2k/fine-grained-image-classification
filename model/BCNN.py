import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


n_features_1 = 2048  # resnet18-2048, resnet34-2048, resnet50-8192
n_features_2 = 2048
fmap_size = 7


def setup_resnet(fine_tune, model='resnet34', unfreeze_layers=None):
    if model == 'resnet18':
        resnet = models.resnet18(pretrained=True)
    elif model == 'resnet34':
        resnet = models.resnet34(pretrained=True)
    elif model == 'resnet50':
        resnet = models.resnet50(pretrained=True)

    # freezing parameters
    if not fine_tune:
        for param in resnet.parameters():
            param.requires_grad = False

    elif unfreeze_layers is not None:
        for name, param in resnet.named_parameters():
            layer_name = str(name).split('.')
            if layer_name[0] in unfreeze_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        for param in resnet.parameters():
            param.requires_grad = True
    
    layers = list(resnet.children())[:-2]
    features = nn.Sequential(*layers).cuda()

    return features


class BCNN(nn.Module):
    # TODO: modify resnet, add second seperate resnet
    def __init__(self, fine_tune=False, n_classes=100):
        
        super(BCNN, self).__init__()
        
        self.resnet = setup_resnet(fine_tune, model='resnet18', unfreeze_layers=['layer3'])
        self.resnet_2 = setup_resnet(fine_tune, model='resnet34', unfreeze_layers=['layer4'])

        self.fc = nn.Linear(n_features_1 * n_features_2, n_classes)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize the fc layers.
        nn.init.xavier_normal_(self.fc.weight.data)
        
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
        
    def forward(self, x):
        
        ## X: bs, 3, 224, 224
        ## N = bs
        N = x.size()[0]
        
        ## x : bs, 1024, 14, 14
        x_fa = self.resnet(x)
        x_fb = self.resnet_2(x)
        
        # bs, (1024 * 196) matmul (196 * 1024)
        x_fa = x_fa.view(N, n_features_1, fmap_size ** 2)
        x_fb = x_fb.view(N, n_features_2, fmap_size ** 2)
        x_fa = self.dropout(x_fa)
        x_fb = self.dropout(x_fb)
        
        # Batch matrix multiplication
        x = torch.bmm(x_fa, torch.transpose(x_fb, 1, 2))/ (fmap_size ** 2) 

        x = x.view(N, n_features_1 * n_features_2)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


if __name__ == '__main__':
    model = BCNN()
    print(model)