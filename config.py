import os
import importlib
import utils


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
fine_tuned_path = os.path.join(trained_dir, 'fine_tuned_{}.h5')
weights_top_model_path = os.path.join(trained_dir, 'weights_top_model_{}.h5')
weights_fine_tuned_path = os.path.join(trained_dir, 'weights_fine_tuned_{}.h5')

target_size_dict = {
    'inception_v3': (299, 299),
    'mobilenet': (224, 224),
    'resnet50': (224, 224),
    'vgg16': (224, 224),
    'vgg19': (224, 224),
    'xception': (299, 299)
}

pretrained_model_dict = {
    'inception_v3': 'InceptionV3',
    'mobilenet': 'MobileNet',
    'resnet50': 'ResNet50',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
    'xception': 'Xception'
}


def set_dirs():
    """
    """
    global trained_dir, precomputed_dir
    utils.create_dir(trained_dir)
    utils.create_dir(precomputed_dir)


def get_input_shape(model):
    global target_size_dict
    if utils.get_keras_backend_name() == 'theano':
        return (3,) + target_size_dict[model]
    else:
        return target_size_dict[model] + (3,)


def get_pretrained_model(model, *args, **kwargs):
    """
    Return pre-trained model instance

    Parameters
    ----------
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'vgg16',
            'vgg19',
            'xception'
    *args: positioned arguments passed to pre-trained model class
    **kwargs: key-word arguments passed to pre-trained model class
    """
    module = importlib.import_module('keras.applications.{}'.format(model))
    model_class = getattr(module, pretrained_model_dict[model])
    return model_class(*args, **kwargs)


def set_classes_from_train_dir():
    """
    Set classes based on sub-directories in train directory
    """
    global classes, train_dir
    classes = sorted([subf for subf in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, subf))])


def get_top_model_weights_path(model):
    return weights_top_model_path.format(model)


def get_fine_tuned_weights_path(model):
    return weights_fine_tuned_path.format(model)


def get_top_model_path(model):
    return top_model_path.format(model)


def get_fine_tuned_path(model):
    return fine_tuned_path.format(model)


set_dirs()
if not classes:
    set_classes_from_train_dir()
