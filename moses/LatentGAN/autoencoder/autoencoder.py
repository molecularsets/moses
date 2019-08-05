from ddc_pub.ddc_pub import ddc_v3 as ddc
import os

DEFAULT_MODEL_VERSION = '9784435'


def load_model(model_version=None):
    # Import model
    if model_version is None:
        model_version = DEFAULT_MODEL_VERSION
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_version)
    model = ddc.DDC(model_name=path)

    return model
