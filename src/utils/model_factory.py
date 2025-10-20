from src.models.models import (
    SingleFrameModel,
)  # , LateFusionModel, EarlyFusionModel, C3D


def select_model(model_name: str, num_classes: int):
    """
    Velger og instansierer den korrekte modellen basert på navn.
    """
    if model_name == "per_frame_agg":
        # Instansier Single-Frame Aggregation klassen
        return SingleFrameModel(num_classes)
    # elif model_name == 'late_fusion':
    #     # Instansier Late Fusion klassen
    #     return LateFusionModel(num_classes)
    # elif model_name == 'early_fusion':
    #     # Instansier Early Fusion klassen
    #     return EarlyFusionModel(num_classes)
    # elif model_name == 'c3d':
    #     # Instansier 3D CNN klassen
    #     return C3D(num_classes)
    else:
        raise ValueError(f"Ukjent modell: {model_name}")
