from model.model import CBGANModel

def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = CBGANModel
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance