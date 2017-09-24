import os
import errno
from keras import backend as K


abspath = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(abspath, 'data')
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
trained_dir = os.path.join(abspath, 'trained')
precomputed_dir = os.path.join(abspath, 'precomputed')

model = 'resnet50'
classes = []

top_model_path = os.path.join(trained_dir, 'top_model_{}.h5')
transfer_model_path = os.path.join(trained_dir, 'transfer_model_{}.h5')
weights_top_model_path = os.path.join(trained_dir, 'weights_top_model_{}.h5')
weights_transfer_model_path = os.path.join(
    trained_dir, 'weights_transfer_model_{}.h5')

x_train_path = os.path.join(precomputed_dir, 'x_train_{}.h5')
x_valid_path = os.path.join(precomputed_dir, 'x_valid_{}.h5')
x_test_path = os.path.join(precomputed_dir, 'x_test_{}.h5')
y_train_path = os.path.join(precomputed_dir, 'y_train.h5')
y_valid_path = os.path.join(precomputed_dir, 'y_valid.h5')
y_test_path = os.path.join(precomputed_dir, 'y_test.h5')
bf_train_path = os.path.join(
    precomputed_dir, 'bottleneck_features_train_{}.h5')
bf_valid_path = os.path.join(
    precomputed_dir, 'bottleneck_features_valid_{}.h5')
bf_test_path = os.path.join(
    precomputed_dir, 'bottleneck_features_test_{}.h5')

pretrained_model_list = {'inception_v3', 'mobilenet', 'resnet50', 'resnet101',
                         'resnet152', 'vgg16', 'vgg19', 'xception'}

target_size_dict = {
    'inception_v3': (299, 299),
    'mobilenet': (224, 224),
    'resnet50': (224, 224),
    'resnet101': (224, 224),
    'resnet152': (224, 224),
    'vgg16': (224, 224),
    'vgg19': (224, 224),
    'xception': (299, 299)
}

pretrained_model_dict = {
    'inception_v3': 'InceptionV3',
    'mobilenet': 'MobileNet',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
    'xception': 'Xception'
}


def create_dir(dirpath):
    """
    Create a directory if it does not exist
    """
    try:
        os.makedirs(os.path.expanduser(dirpath))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        elif not os.path.isdir(dirpath):
            print(('Warning: `' + dirpath + '` already exists '
                   'and is not a directory!'))


def set_dirs():
    """
    """
    create_dir(trained_dir)
    create_dir(precomputed_dir)


def set_classes_from_train_dir():
    """
    Set classes based on sub-directories in train directory
    """
    global classes
    classes = sorted([subf for subf in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, subf))])


def get_top_model_weights_path(model):
    return weights_top_model_path.format(model)


def get_transfer_model_weights_path(model):
    return weights_transfer_model_path.format(model)


def get_top_model_path(model):
    return top_model_path.format(model)


def get_transfer_model_path(model):
    return transfer_model_path.format(model)


def get_x_train_path(model):
    return x_train_path.format(model)


def get_x_valid_path(model):
    return x_valid_path.format(model)


def get_x_test_path(model):
    return x_test_path.format(model)


def get_bf_train_path(model):
    return bf_train_path.format(model)


def get_bf_valid_path(model):
    return bf_valid_path.format(model)


def get_bf_test_path(model):
    return bf_test_path.format(model)


def set_image_format():
    if K.backend() == 'theano':
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')


set_dirs()
if not classes:
    set_classes_from_train_dir()
set_image_format()
