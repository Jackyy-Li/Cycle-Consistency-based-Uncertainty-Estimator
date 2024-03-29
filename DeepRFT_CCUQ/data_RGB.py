from dataset_RGB import *


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(inp_dir, tar_dir, img_options):
    # assert os.path.exists(rgb_dir)
    return DataLoaderTest(inp_dir, tar_dir, img_options)
