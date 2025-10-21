from src.models.models import (
    SingleFrameModel,
    LateFusionModel,
    EarlyFusionModel,
    C3DModel,
)


def select_model(model_name: str, num_classes: int):
    """
    Velger og instansierer den korrekte modellen basert på navn.
    """
    if model_name == "per_frame":
        return SingleFrameModel(num_classes)
    elif model_name == "late_fusion":
        return LateFusionModel(num_classes)
    elif model_name == "early_fusion":
        return EarlyFusionModel(num_classes)
    elif model_name == "3d_cnn":
        return C3DModel(num_classes)
    else:
        raise ValueError(f"Ukjent modell: {model_name}")
