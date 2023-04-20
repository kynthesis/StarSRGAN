from copy import deepcopy
from starsrgan.utils.registry import MODEL_REGISTRY
from starsrgan.utils.logger import get_root_logger


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt["model_type"])(opt)
    logger = get_root_logger()
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    return model
