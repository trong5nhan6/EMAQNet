import torch
from torchvision.models import mobilenet_v3_small, resnet18, mobilenet_v2
import torch.nn as nn

def get_model(device, network,tabular_switch,S5p_switch, checkpoint=None):

    #PARAMETERS
    tabular_input_count = 14
    prediction_count = 3
    S2_num_features = 640
    S5p_num_features = 128
    tabular_features = 32
    head_features = 96

    #SENTINEL2 BACKBONE
    assert network in ["mobilenetv3small","resnet18"]
    if network == "mobilenetv3small":
        backbone_S2 = mobilenet_v3_small(pretrained=checkpoint, num_classes=1000)
        backbone_S2.features[0][0] = nn.Conv2d(12, 16, 3, 1, 1)
        backbone_S2.classifier[3] = nn.Linear(1024,S2_num_features)
    elif network == "resnet18":
        backbone_S2 = resnet18(pretrained=checkpoint, num_classes=1000)
        backbone_S2.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=64, kernel_size=(3, 3), stride=(2, 2),padding=(3, 3), bias=False)
        backbone_S2.fc = nn.Linear(512, S2_num_features)

    backbone_S5P = mobilenet_v2(pretrained=False)
    backbone_S5P.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # sửa lại cho 1 kênh
    backbone_S5P.classifier = nn.Sequential(
        nn.Linear(backbone_S5P.last_channel, S5p_num_features),  # gán về số chiều bạn cần
        nn.ReLU(),
        nn.Dropout(0.3)
)

    # #SENTINEL5P and TABULAR BACKBONE
    # backbone_S5P = nn.Sequential(nn.Conv2d(1, 10, 3),
    #                           nn.ReLU(),
    #                           nn.MaxPool2d(3),
    #                           nn.Conv2d(10, 15, 5),
    #                           nn.ReLU(),
    #                           nn.MaxPool2d(3),
    #                           nn.Flatten(),
    #                           nn.Linear(1815, S5p_num_features))
    
    backbone_tabular = nn.Sequential(
                    nn.Linear(tabular_input_count, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, tabular_features))

    #REGRESSION HEADS
    head_1 = nn.Sequential(
            nn.Linear(S2_num_features+S5p_num_features, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192, head_features))
    
    head_2 = nn.Sequential(
        nn.Linear(S2_num_features + S5p_num_features, 384),
        nn.BatchNorm1d(384),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(384, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(32, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(16, prediction_count))
    mixer_1 = nn.Sequential(
        nn.Linear(head_features+tabular_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, prediction_count))

    #SWITCHES
    if S5p_switch == True: #S5p_switch == True:
        if tabular_switch == True: # use s2, s5p and tabular
            regression_model = RegressionHead_1(backbone_S2, backbone_S5P, head_1, backbone_tabular,mixer_1)
        else:                       # use s2, s5p
            regression_model = RegressionHead_2(backbone_S2, backbone_S5P, head_2)

    # print("Displaying model architecture")
    # print(regression_model)

    return regression_model

class RegressionHead_1(nn.Module): # s2, s5p, and tabular
    def __init__(self, backbone_S2, backbone_S5P, head_1,backbone_tabular, mixer_1):
        super(RegressionHead_1, self).__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head_1 = head_1
        self.backbone_tabular = backbone_tabular
        self.mixer_1 = mixer_1
    def forward(self, x):
        #get
        img = x.get("img")
        s5p = x.get("s5p")
        tabular = x.get("tabular")
        #backbones
        img = self.backbone_S2(img)
        s5p = self.backbone_S5P(s5p)
        #satellites
        x = torch.cat((img, s5p), dim=1)
        x = self.head_1(x)
        #satellites+tabular
        tabular = self.backbone_tabular(tabular)
        x = torch.cat((x,tabular),dim=1)
        out = self.mixer_1(x)
        out_1 = out[:,0]
        out_2 = out[:,1]
        out_3 = out[:,2]
        return out_1,out_2,out_3

class RegressionHead_2(nn.Module): # use s2, s5p
    def __init__(self, backbone_S2, backbone_S5P, head_2):
        super(RegressionHead_2, self).__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head_2 = head_2
    def forward(self, x):
        #get
        img = x.get("img")
        s5p = x.get("s5p")
        #backbones
        img = self.backbone_S2(img)
        s5p = self.backbone_S5P(s5p)
        #satellites
        x = torch.cat((img, s5p), dim=1)
        out = self.head_2(x)
        out_1 = out[:,0]
        out_2 = out[:,1]
        out_3 = out[:,2]
        return out_1,out_2,out_3