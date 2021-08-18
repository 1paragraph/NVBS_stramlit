import torch
import torch.nn as nn

class Class_Net(nn.Module):
    def __init__(self, backbone):
        super(Class_Net, self).__init__()
        self.backbone = backbone
#         self._digit_length = nn.Sequential(nn.Linear(1000, 7))
        self.way_out = nn.Sequential(nn.Linear(1000, 10))
        
    def forward(self, x):
        x = self.backbone(x)
        way_out = self.way_out(x)

        return way_out
    
class Crop_Net(nn.Module):
    def __init__(self, backbone):
        super(Crop_Net, self).__init__()
        self.backbone = backbone
#         self._digit_length = nn.Sequential(nn.Linear(1000, 7))
        self._digit1 = nn.Sequential(nn.Linear(1000, 4))
        self._digit2 = nn.Sequential(nn.Linear(1000, 4))
        self._digit3 = nn.Sequential(nn.Linear(1000, 4))
        self._digit4 = nn.Sequential(nn.Linear(1000, 4))
        self._digit5 = nn.Sequential(nn.Linear(1000, 4))
        self._digit6 = nn.Sequential(nn.Linear(1000, 4))
        
    def forward(self, x):
        x = self.backbone(x)
        digit1_logits = self._digit1(x).unsqueeze(0)
        digit2_logits = self._digit2(x).unsqueeze(0)
        digit3_logits = self._digit3(x).unsqueeze(0)
        digit4_logits = self._digit4(x).unsqueeze(0)
        digit5_logits = self._digit5(x).unsqueeze(0)
        digit6_logits = self._digit6(x).unsqueeze(0)
        
        out = torch.cat([digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits, digit6_logits])#.log_softmax(2)

        return out
