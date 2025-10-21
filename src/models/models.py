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


class LateFusionModel(nn.Module):
    """
    Late fusion for video:
      - Kjører en 2D-ResNet50 på hver frame
      - Head på features per frame
      - Gjennomsnitt (mean) av logits over tid -> [B, K]

    Inndata støttes i to varianter:
      - list/tuple med T tensors [B,C,H,W]
      - stacket tensor [B,C,T,H,W].
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
        weights=ResNet50_Weights.IMAGENET1K_V1,
    ):
        super().__init__()

        # Backbone = ResNet50 uten siste FC
        m = resnet50(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()

        # Frys/ikke frys backbone
        for p in m.parameters():
            p.requires_grad = not freeze_backbone

        self.backbone = m
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, num_classes),
        )

    def _to_BTCHW(self, x):
        """
        Normaliser input til [B*T, C, H, W] og returner B, T.
        """
        if isinstance(x, (list, tuple)):
            # liste med T tensors [B,C,H,W]
            assert len(x) > 0, "Tom sekvens av frames"
            T = len(x)
            B, C, H, W = x[0].shape
            x_bt = torch.stack(x, dim=1).reshape(
                B * T, C, H, W
            )  # [B,T,C,H,W] -> [B*T,C,H,W]
            return x_bt, B, T
        else:
            # stacket [B,C,T,H,W] -> [B*T,C,H,W]
            assert x.dim() == 5 and x.size(2) >= 1, "Forventet [B,C,T,H,W]"
            B, C, T, H, W = x.shape
            x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            return x_bt, B, T

    def forward(self, x):
        """
        Returnerer video-logits [B, num_classes].
        """
        x_bt, B, T = self._to_BTCHW(x)  # [B*T,C,H,W]
        feats_bt = self.backbone(x_bt)  # [B*T, F]
        logits_bt = self.head(feats_bt)  # [B*T, K]
        logits_bt = logits_bt.view(B, T, -1)  # [B, T, K]
        return logits_bt.mean(dim=1)  # late fusion (avg) -> [B, K]
