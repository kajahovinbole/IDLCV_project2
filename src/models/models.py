import torch
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

class EarlyFusionModel(nn.Module):
    """
    Early fusion for video:
      - Konkatenér T frames på kanalaksen: [B,C,T,H,W] -> [B, C*T, H, W]
      - 2D-ResNet50 (første conv utvides fra 3 -> 3*T inngangskanaler)
      - Ett logits-output per video: [B, K]

    Inndata støttes i to varianter:
      - list/tuple med T tensors [B,C,H,W]
      - stacket tensor [B,C,T,H,W]

    Viktig: Bruk samme per-frame-transform som for SingleFrameModel.
            (Dvs. normaliser hvert frame likt før stakking)
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
        weights=ResNet50_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        m = resnet50(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()

        # Frys/ikke frys backbone (vi setter requires_grad etter at conv1 ev. er utvidet)
        self.backbone = m
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, num_classes),
        )
        self._frozen = freeze_backbone  # brukes etter vi vet T

    # ---- helpers ----
    def _to_BCTHW(self, x):
        """
        Normaliser input til [B,C,T,H,W] og returner også B,C,T,H,W.
        """
        if isinstance(x, (list, tuple)):
            assert len(x) > 0, "Tom sekvens av frames"
            # x[t]: [B,C,H,W] — stack til [B,T,C,H,W] -> [B,C,T,H,W]
            B, C, H, W = x[0].shape
            T = len(x)
            x_btc_hw = torch.stack(x, dim=1)           # [B,T,C,H,W]
            x_bct_hw = x_btc_hw.permute(0, 2, 1, 3, 4) # [B,C,T,H,W]
            return x_bct_hw, (B, C, T, H, W)
        else:
            # Forventet [B,C,T,H,W]
            assert x.dim() == 5 and x.size(2) >= 1, "Forventet [B,C,T,H,W]"
            B, C, T, H, W = x.shape
            return x, (B, C, T, H, W)

    def _ensure_conv1_in_channels(self, T: int, device, dtype):
        """
        Sørg for at første conv tar 3*T kanaler (early fusion).
        Init: repeter IMAGENET-vektene over tidsdim og skaler med 1/T (gjennomsnitt).
        Kalles første gang i forward når vi vet T.
        """
        conv1 = self.backbone.conv1
        in_ch = conv1.in_channels
        if in_ch == 3 * T:
            return  # alt ok
        assert in_ch == 3, f"Forventet opprinnelig conv1.in_channels=3, fikk {in_ch}"

        # Lag en ny conv med 3*T inngangskanaler
        new_conv = nn.Conv2d(
            in_channels=3 * T,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False,
        ).to(device=device, dtype=dtype)

        # Init-vekter: gjennomsnitt av RGB-vekter repetert T ganger
        with torch.no_grad():
            # conv1.weight: [out_ch, 3, kH, kW]
            w = conv1.weight.to(device=device, dtype=dtype)
            w_rep = w.repeat(1, T, 1, 1)  # [out_ch, 3*T, kH, kW]
            # Del på T slik at energien per utkanal holder seg omtrent lik
            new_conv.weight.copy_(w_rep / T)

        self.backbone.conv1 = new_conv

        # Re-frys/backbone om ønsket
        for p in self.backbone.parameters():
            p.requires_grad = not self._frozen

    # ---- forward ----
    def forward(self, x):
        """
        x: [B,C,T,H,W] eller list[T x [B,C,H,W]]
        Returnerer: [B, num_classes]
        """
        x, (B, C, T, H, W) = self._to_BCTHW(x)           # [B,C,T,H,W]
        self._ensure_conv1_in_channels(T, x.device, x.dtype)

        # Early fusion: kanal-konkatenasjon
        x_bct_hw = x
        x_bct_hw = x_bct_hw.contiguous()
        x_bct_hw = x_bct_hw.view(B, C * T, H, W)         # [B, C*T, H, W]

        feats = self.backbone(x_bct_hw)                  # [B, F]
        logits = self.head(feats)                        # [B, K]
        return logits  