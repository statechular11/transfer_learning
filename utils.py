import os
import sys
import shutil
import errno
import imghdr
import requests
import importlib
import h5py
import config
from io import BytesIO
from collections import namedtuple
from keras import backend as K


# define constants
IMAGE_EXTENSIONS = {
    'rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff',
    'rast', 'xbm', 'jpg', 'jpeg', 'bmp', 'png'
}


def remove_file(filepath):
    """
    Remove a file if exists
    """
    try:
        os.remove(os.path.expanduser(filepath))
    except OSError as exception:
        if exception.errno != errno.ENOENT:
            raise


def ceildiv(dividend, divisor):
    """ceiling-division for two integers
    """
    return -(-dividend // divisor)


def file_of_extensions(filepath, ext_list):
    """
    Check the extension of a file is one of given list of extensions
    """
    _, ext = os.path.splitext(filepath)
    return (ext[1:]).lower() in ext_list


def files_under_dir(dirpath, followlinks=False):
    """Return a list of all files in a given directory and its subdirectories
    """
    abs_path = os.path.abspath(os.path.expanduser(dirpath))
    files = [os.path.join(root, file)
             for root, _, files in os.walk(abs_path, followlinks=followlinks)
             for file in files]
    return sorted(files)


def files_under_subdirs(dirpath, subdirs=None, followlinks=False):
    """
    Return a list of all files in a given directory and specified subdirs
    """
    abs_path = os.path.abspath(os.path.expanduser(dirpath))
    if not subdirs:
        subdirs = os.listdir(abs_path)
    subdirs = [os.path.basename(subdir) for subdir in subdirs]
    files = [file for subdir in subdirs
             for file in files_under_dir(os.path.join(abs_path, subdir),
                                         followlinks=followlinks)]
    return sorted(files)


def images_under_dir(dirpath,
                     examine_by='extension',
                     ext_list=config.IMAGE_EXTENSIONS,
                     followlinks=False):
    """
    Return a list of image files in a given dir and its subdirs

    Parameters
    ----------
    dirpath: string or unicode
        path to the directory
    examine_by: string (default='extension')
        method of examining image file, either 'content' or 'extension'
        If 'content', `imghdr.what()` is used
        If 'extension', the file extension is compared against a list of common
        image extensions
    ext_list: list, optional (used only when examine_by='extension')
        If examine_by='extension', the file extension is compared against
        ext_list
    """
    files = files_under_dir(dirpath, followlinks=followlinks)
    assert examine_by in ['content', 'extension']
    if examine_by == 'content':
        images = [file for file in files if imghdr.what(file)]
    else:
        images = [file for file in files if file_of_extensions(file, ext_list)]
    return images


def images_under_subdirs(dirpath,
                         subdirs=None,
                         examine_by='extension',
                         ext_list=config.IMAGE_EXTENSIONS,
                         followlinks=False):
    """
    Return a list of image files in a given dir and specified subdirs
    """
    files = files_under_subdirs(dirpath, subdirs, followlinks=followlinks)
    assert examine_by in ['content', 'extension']
    if examine_by == 'content':
        images = [file for file in files if imghdr.what(file)]
    else:
        images = [file for file in files if file_of_extensions(file, ext_list)]
    return images


def load_h5file(filepath):
    filepath = os.path.abspath(filepath)
    assert os.path.exists(filepath)
    _, file_extension = os.path.splitext(filepath)
    assert file_extension == '.h5'
    with h5py.File(filepath, 'r') as hf:
        keys = list(hf.keys())
        assert len(keys) >= 1
        if len(keys) == 1:
            return hf[keys[0]][:]
        result = {key: hf[key][:] for key in keys}
        return result


def is_keras_pretrained_model(model):
    """
    Check if a model is on the keras pre-trained model list, i.e.,
    'inception_v3', 'mobilenet', 'resnet50', 'resnet152', 'vgg16', 'vgg19',
    'xception'

    Parameters
    ----------
    model: string
        name of a model

    Returns
    -------
        boolean. `True` is the model is on the keras pre-trained model list and
        `False` otherwise
    """
    return model in config.pretrained_model_list


def get_input_shape(model=None, data_format=None):
    '''Get correct input shape for pre-trained models
    '''
    if model is None:
        model = config.model
    assert is_keras_pretrained_model(model)
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    if data_format == 'channels_first':
        return (3,) + config.target_size_dict[model]
    else:
        return config.target_size_dict[model] + (3,)


def get_pretrained_model(model=None, *args, **kwargs):
    """
    Return pre-trained model instance

    Parameters
    ----------
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101'
            'resnet152'
            'vgg16',
            'vgg19',
            'xception'
    *args: positioned arguments passed to pre-trained model class
    **kwargs: key-word arguments passed to pre-trained model class
    """
    if model is None:
        model = config.model
    assert is_keras_pretrained_model(model)
    if model in {'resnet101', 'resnet152'}:
        module = importlib.import_module('resnet')
    else:
        module = importlib.import_module('keras.applications.{}'.format(model))
    model_class = getattr(module, config.pretrained_model_dict[model])
    return model_class(*args, **kwargs)


def url2file(url):
    """
    Read a URL and return a file(-like) object, which can be further provided
    to `keras.preprocessing.image.load_img()`
    """
    try:
        response = requests.get(url)
        file_obj = BytesIO(response.content)
        return file_obj
    except requests.exceptions.RequestException as e:
        print('Can read {}'.format(url))
        raise e
        sys.exit(1)


def move_files_between_dirs(src, dst):
    """
    Move all the files under origin directory to destination directory
    """
    config.create_dir(dst)
    for filename in os.listdir(src):
        shutil.move(os.path.join(src, filename), os.path.join(dst, filename))


def copy_files_between_dirs(src, dst):
    """
    Copy all the files under origin directory to destination directory
    """
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        return
    for filename in os.listdir(src):
        try:
            shutil.copytree(os.path.join(src, filename),
                            os.path.join(dst, filename))
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(os.path.join(src, filename),
                            os.path.join(dst, filename))
            else:
                raise


def decode_predictions(preds, top=3, classes=None):
    """
    Decodes the prediction(s) of an image classification model

    Parameters
    ----------
    preds: numpy array
        a batch of predictions from the trained classification model
    top: int
        how many top guesses to return

    Returns
    -------
        a list of lists of numedtuples, `(label, score)`. One list of
        numedtuples per sample in batch input
    """
    if classes is None:
        classes = config.classes
    pred_tuple_proto = namedtuple('Predictions', ['label', 'score'])
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [
            pred_tuple_proto(label=classes[idx], score=pred[idx])
            for idx in top_indices
        ]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results
