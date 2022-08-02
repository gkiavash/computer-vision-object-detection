import os

OBJECT_TO_DETECT = 'boat'

ORIG_BASE_PATH = "dataset"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

TEST_BASE_PATH = "dataset_test"
TEST_IMAGES = os.path.sep.join([TEST_BASE_PATH, "images"])
TEST_ANNOTS = os.path.sep.join([TEST_BASE_PATH, "annotations"])

BASE_PATH = "training_data"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "boat"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_boat"])

BASE_RESULT_PATH = 'results'
RESULT_IMAGE_PATH = os.path.sep.join([BASE_RESULT_PATH, "images"])

RESULT_MODEL_PATH_MobileNet = os.path.sep.join([BASE_RESULT_PATH, "model_mobilenet"])
RESULT_MODEL_PATH_CNN = os.path.sep.join([BASE_RESULT_PATH, "model_cnn"])
RESULT_MODEL_PATH_Inception = os.path.sep.join([BASE_RESULT_PATH, "model_inception"])
RESULT_MODEL_PATH_EfficientNetB0 = os.path.sep.join([BASE_RESULT_PATH, "model_model_EfficientNetB0"])

ENCODER_PATH = RESULT_MODEL_PATH_MobileNet + "/label_encoder_boat.pickle"

MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUT_DIMS = (224, 224)
INPUT_DIMS_3D = (224, 224, 3)


MIN_PROBA = 0.1
INIT_LR = 0.01
EPOCHS = 20
BS = 128

train_test_split_proportion = 0.2
aug_rotation_range = 90
aug_zoom_range = 0.05
