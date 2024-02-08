import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fusion_model import *

class MultiModalModel(nn.Module):
    def __init__(self,video_model,audio_model,num_class = 101,fusion_method='sum'):
        super(MultiModalModel,self).__init__()
        self.audio_net = audio_model
        self.visual_net = video_model

        fusion = fusion_method
        if fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=num_class)
        elif fusion == 'orth':
            self.fusion_module = OrthFusion(output_dim=num_class) 
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        print("Using {} Fusion!!".format(fusion))
    
    def forward(self,visual,audio):

        v = self.visual_net(visual)
        a = self.audio_net(audio)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        v, a, out = self.fusion_module(v, a)

        return v, a, out