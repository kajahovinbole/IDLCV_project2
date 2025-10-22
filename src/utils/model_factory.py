from src.models.models import (
    SingleFrameModel,
    LateFusionModel,
    EarlyFusionModel,
    C3DModel,
)

from config import DROPOUT, FREEZE_BACKBONE, TRAINABLE_BLOCKS


def select_model(model_name: str, num_classes: int):
    """
    Velger og instansierer den korrekte modellen basert på navn.
    """
    if model_name == "per_frame":
        return SingleFrameModel(
            num_classes,
            dropout_p=DROPOUT,
            freeze_backbone=FREEZE_BACKBONE,
            trainable_blocks=TRAINABLE_BLOCKS,
        )
    elif model_name == "late_fusion":
        return LateFusionModel(
            num_classes, dropout_p=DROPOUT, freeze_backbone=FREEZE_BACKBONE
        )
    elif model_name == "early_fusion":
        return EarlyFusionModel(
            num_classes, dropout_p=DROPOUT, freeze_backbone=FREEZE_BACKBONE
        )
    elif model_name == "3d_cnn":
        return C3DModel(num_classes, dropout_p=DROPOUT)
    else:
        raise ValueError(f"Ukjent modell: {model_name}")
