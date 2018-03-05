MODEL_DATA_IGNORE_STRING = "ModelData"


def get_model_filename(model_name):
    return "{}_{}".format(MODEL_DATA_IGNORE_STRING, model_name)
