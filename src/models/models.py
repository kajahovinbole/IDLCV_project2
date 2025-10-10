import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SingleFrameModel(nn.Module):
    """
    Pretrained ResNet50 som klassifiserer ett bilde [B,C,H,W].
    Bruk riktig transform (se get_single_frame_transform) før du mater inn.
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        trainable_blocks: int = 0,  # 0: frys alt, 1: unfreeze layer4, 2: layer3+4
        weights=ResNet50_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        m = resnet50(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()  # hent features, ikke 1000-logits

        # frys alt
        for p in m.parameters():
            p.requires_grad = not freeze_backbone

        # ev. tine opp siste blokker
        # if freeze_backbone and trainable_blocks >= 1:
        #     for p in m.layer4.parameters(): p.requires_grad = True
        # if freeze_backbone and trainable_blocks >= 2:
        #     for p in m.layer3.parameters(): p.requires_grad = True

        self.backbone = m
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, num_classes))

    def forward(self, x):  # x må være transformert/normalisert for ResNet
        f = self.backbone(x)  # [B, F]
        return self.head(f)  # [B, num_classes]


# class EarlyFusionModel(nn.Module):
#     def __init__(self, T=10, num_classes=NUM_CLASSES):
#         super(EarlyFusionModel, self).__init__()

#         # Første lag må håndtere T*3 inngangskanaler
#         self.conv1 = nn.Conv2d(3 * T, 64, kernel_size=5, padding=2)
#         self.relu = nn.ReLU(inplace=True)

#         # Resten av nettverket er standard 2D CNN
#         self.features = nn.Sequential(
#             # ... Andre 2D lag
#         )
#         self.classifier = nn.Sequential(
#             # ... FC lag
#             nn.Linear(..., num_classes)
#         )

#     def forward(self, x):
#         # x har form (Batch, T, C, H, W)
#         # Stack rammene: Gjør om til (Batch, C*T, H, W)
#         B, T, C, H, W = x.size()
#         x = x.view(B, T * C, H, W) # Flatten T og C

#         x = self.relu(self.conv1(x))
#         # ... resten av forward pass som en standard 2D CNN

#         return x


# class LateFusionModel(nn.Module):
#     def __init__(self, T=10, num_classes=NUM_CLASSES):
#         super(LateFusionModel, self).__init__()
#         self.T = T

#         # Definer en felles 2D feature extractor (uten det siste FC-laget)
#         self.base_cnn = SingleFrameModel(num_classes=NUM_CLASSES) # Gjenbruk SingleFrameModel's feature del

#         # Erstatt det siste laget for å få en vektor før klassifisering
#         self.base_cnn.classifier = self.base_cnn.classifier[:-1]
#         self.feature_dim = 4096 # Gitt at forrige lag produserte 4096 features

#         # Fusjons- og klassifiseringslag
#         # Slå sammen (concatenate) alle trekkvektorer (T * feature_dim)
#         self.fusion_classifier = nn.Sequential(
#             nn.Linear(self.T * self.feature_dim, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, num_classes)
#         )

#     def forward(self, x):
#         # x har form (Batch, T, C, H, W)
#         B, T, C, H, W = x.size()

#         features = []
#         for t in range(T):
#             frame = x[:, t, :, :, :] # Velg ramme t: (Batch, C, H, W)
#             feature = self.base_cnn(frame) # Trekk ut feature vektor
#             features.append(feature)

#         # Fusjonering: Konkatenér alle trekkvektorene
#         x_fused = torch.cat(features, dim=1) # (Batch, T * feature_dim)

#         return self.fusion_classifier(x_fused)

# class C3DModel(nn.Module):
#     def __init__(self, num_classes=NUM_CLASSES):
#         super(C3DModel, self).__init__()

#         # Bruk 3D konvolusjonslag
#         self.features = nn.Sequential(
#             # Første lag (f.eks. 3x3x3 kernel)
#             nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), # Eksempel fra C3D [16]
#             # ... Flere 3D lag
#         )

#         # Siste lag (etter 3D pooling/konvolusjon)
#         self.classifier = nn.Sequential(
#             # ... Lineære lag
#             nn.Linear(..., num_classes)
#         )

#     def forward(self, x):
#         # x har form (Batch, C, T, H, W) - Vær obs på rekkefølgen!
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
