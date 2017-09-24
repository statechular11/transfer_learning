import os
import importlib
import numpy as np
import h5py
import config
import utils
from functools import partial
from sklearn.utils import check_random_state
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical


def preprocess_an_image(image, model=None):
    """
    Wrapper around `keras.applications.{model}.preprocess_input()`

    Parameters
    ----------
    image: a 3D/4D numpy array
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101',
            'resnet152',
            'vgg16',
            'vgg19',
            'xception'

    Returns
    -------
    A 3D/4D numpy array (preprocessed image)
    """
    if model is None:
        model = config.model
    assert utils.is_keras_pretrained_model(model) and image.ndim in {3, 4}
    if model in {'resnet101', 'resnet152'}:
        model = 'resnet50'
    module = importlib.import_module('keras.applications.{}'.format(model))
    preprocess_input = module.preprocess_input
    if image.ndim == 3:
        return preprocess_input(np.expand_dims(image, axis=0))[0]
    else:
        return preprocess_input(image)


def preprocess_input_wrapper(model):
    """
    Return a function that does input preprocess for pre-trained model and is
    compatible for use with `keras.preprocessing.image.ImageDataGenerator`'s
    `preprocessing_function` argument

    Parameters
    ----------
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101',
            'resnet152',
            'vgg16',
            'vgg19',
            'xception'
    """
    return partial(preprocess_an_image, model=model)


def path_to_tensor(image_path, target_size, grayscale=False, data_format=None):
    """
    Read an image from its path, resize it to a specified size (height, width),
    and return a numpy array that is ready to be passed to the `predict` method
    of a trained model

    Parameters
    ----------
    image_path: string
        the path of an image
    target_size: tuple/list
        (height, width) of the image
    grayscale: bool
        whether to load the image as grayscale
    data_format: str
        one of `channels_first`, `channels_last`

    Returns
    -------
        a numpy array that is to be readily passed to the `predict` method of
        a trained model
    """
    assert os.path.exists(os.path.abspath(image_path))
    if data_format is None:
        data_format = K.image_data_format()
    image = load_img(image_path, grayscale=grayscale, target_size=target_size)
    tensor = img_to_array(image, data_format=data_format)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def get_x_from_path(model=None,
                    container_path=None,
                    classes=None,
                    save=False,
                    filename=None,
                    verbose=False):
    """
    """
    if model is None:
        model = config.model
    assert utils.is_keras_pretrained_model(model)
    if container_path is None:
        container_path = config.train_dir
    imagepaths = utils.images_under_subdirs(container_path, subdirs=classes)
    tensor_list = []
    target_size = config.target_size_dict[model]
    if verbose:
        print('Started: images -> tensors')
    for path in imagepaths:
        tensor = path_to_tensor(path, target_size=target_size)
        tensor_list.append(tensor)
    preprocess_fun = preprocess_input_wrapper(model)
    tensors = np.vstack(tensor_list)
    tensors = preprocess_fun(tensors)
    if verbose:
        print('Finished: images -> tensors')
    if save:
        if not filename:
            filename = 'x_{}.h5'.format(config.model)
        filepath = os.path.join(config.precomputed_dir, filename)
        utils.remove_file(filepath)
        if verbose:
            print('Started saving {}'.format(filename))
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('data', data=tensors)
        if verbose:
            print('Finished saving {}'.format(filename))
    else:
        return tensors


def get_x_from_path_train(model=None,
                          classes=None,
                          save=False,
                          filename=None,
                          verbose=False):
    if model is None:
        model = config.model
    container_path = config.train_dir
    if not filename:
        filename = config.get_x_train_path(model)
    return get_x_from_path(
        model=model,
        container_path=container_path,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def get_x_from_path_valid(model=None,
                          classes=None,
                          save=False,
                          filename=None,
                          verbose=False):
    if model is None:
        model = config.model
    container_path = config.valid_dir
    if not filename:
        filename = config.get_x_valid_path(model)
    return get_x_from_path(
        model=model,
        container_path=container_path,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def get_x_from_path_test(model=None,
                         classes=None,
                         save=False,
                         filename=None,
                         verbose=False):
    if model is None:
        model = config.model
    container_path = config.test_dir
    if not filename:
        filename = config.get_x_test_path(model)
    return get_x_from_path(
        model=model,
        container_path=container_path,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def get_bottleneck_features(model=None,
                            source='path',
                            container_path=None,
                            tensor=None,
                            classes=None,
                            save=False,
                            filename=None,
                            verbose=False):
    """Extract bottleneck features

    Parameters
    ----------
    model: string
        pre-trained model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101',
            'resnet152',
            'vgg16',
            'vgg19',
            'xception'
    source: string
        where to extract bottleneck features, either 'path' or 'tensor'
    container_path: string
        if `source='path'`, `container_path` specifies the folder path that
        contains images of all the classes. If `None`, container_path will be
        set to 'path_to_the_module/data/train'
    tensor: numpy array/string
        if `source='tensor'`, `tensor` specifies the tensor from which
        bottleneck features are extracted or the path to the saved tensor file
    classes: tuple/list
        a tuple/list of classes for prediction
    save: boolen
        whether to save the extracted bottleneck features or not
    filename: string
        if `save=True`, specifies the name of the file in which the bottleneck
        features are saved
    verbose: boolean
        verbosity mode
    """
    assert source in {'path', 'tensor'}
    if source == 'path':
        tensors = get_x_from_path(
            model=model,
            container_path=container_path,
            classes=classes,
            save=False,
            verbose=verbose)
    else:
        assert isinstance(tensor, (str, np.ndarray))
        if isinstance(tensor, np.ndarray):
            tensors = tensor
        else:
            assert os.path.exists(tensor)
            tensors = utils.load_h5file(tensor)
    input_shape = utils.get_input_shape(model)
    pretrained_model = utils.get_pretrained_model(
        model,
        include_top=False,
        input_shape=input_shape)
    bottleneck_features = pretrained_model.predict(
        tensors,
        verbose=1 if verbose else 0)
    if save:
        assert filename is not None
        filepath = os.path.join(config.precomputed_dir, filename)
        utils.remove_file(filepath)
        if verbose:
            print('Started saving {}'.format(filename))
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('data', data=bottleneck_features)
        if verbose:
            print('Finished saving {}'.format(filename))
    else:
        return bottleneck_features


def get_bottleneck_features_train(model=None,
                                  source='path',
                                  classes=None,
                                  save=False,
                                  filename=None,
                                  verbose=False):
    if model is None:
        model = config.model
    container_path = config.train_dir
    tensor = config.get_x_train_path(model)
    if not filename:
        filename = config.get_bf_train_path(model)
    return get_bottleneck_features(
        model,
        source=source,
        container_path=container_path,
        tensor=tensor,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def get_bottleneck_features_valid(model=None,
                                  source='path',
                                  classes=None,
                                  save=False,
                                  filename=None,
                                  verbose=False):
    if model is None:
        model = config.model
    container_path = config.valid_dir
    tensor = config.get_x_valid_path(model)
    if not filename:
        filename = config.get_bf_valid_path(model)
    return get_bottleneck_features(
        model,
        source=source,
        container_path=container_path,
        tensor=tensor,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def get_bottleneck_features_test(model=None,
                                 source='path',
                                 classes=None,
                                 save=False,
                                 filename=None,
                                 verbose=False):
    if model is None:
        model = config.model
    container_path = config.test_dir
    tensor = config.get_x_test_path(model)
    if not filename:
        filename = config.get_bf_test_path(model)
    return get_bottleneck_features(
        model,
        source=source,
        container_path=container_path,
        tensor=tensor,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def get_y_from_path(container_path,
                    classes=None,
                    shuffle=False,
                    random_state=0,
                    save=False,
                    filename=None,
                    verbose=False):
    """
    Load y/target/class name for each input image

    Individual samples are assumed to be image files stored a two-level folder
    structure such as the following:

        container_path/
            category_1/
                file_11.jpg
                file_12.jpg
                ...
            category_2/
                file_21.jpg
                file_22.jpg
                ...
            ...
            category_n/
                file_n1.jpg
                file_n2.jpg
                ...

    The folder name of each category is used to be the y/target/class name for
    all the image files stored under

    Parameters
    ----------
    container_path: string or unicode
        Path to the main folder holding one subfolder per category
    shuffle:    bool, optional (default=False)
        Whether or not to shuffle the files
    random_state: int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`
    verbose: boolen
        Verbosity mode

    Returns
    -------
    y/target/class: A 1D numpy array
    """
    targets = []
    num_classes = 0
    if not classes:
        classes = sorted(os.listdir(container_path))
    subfolders = [os.path.join(container_path, subf)
                  for subf in classes
                  if os.path.isdir(os.path.join(container_path, subf))]
    for idx, subf in enumerate(subfolders):
        num_images = len(utils.images_under_dir(subf, examine_by='extension'))
        targets.extend(num_images * [idx])
        if num_images > 0:
            num_classes += 1
    targets = np.array(targets)
    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(targets.shape[0])
        random_state.shuffle(indices)
        targets = targets[indices]
    targets_one_hot_encode = to_categorical(targets, num_classes)
    if save:
        assert filename is not None
        filepath = os.path.join(config.precomputed_dir, filename)
        utils.remove_file(filepath)
        if verbose:
            print('Started saving {}'.format(filename))
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('data', data=targets_one_hot_encode)
        if verbose:
            print('Finished saving {}'.format(filename))
    else:
        return targets_one_hot_encode


def get_y_from_path_train(classes=None,
                          shuffle=False,
                          random_state=0,
                          save=False,
                          filename=None,
                          verbose=False):
    container_path = config.train_dir
    if not filename:
        filename = config.y_train_path
    return get_y_from_path(
        classes=classes,
        container_path=container_path,
        shuffle=shuffle,
        random_state=random_state,
        save=save,
        filename=filename,
        verbose=verbose)


def get_y_from_path_valid(classes=None,
                          shuffle=False,
                          random_state=0,
                          save=False,
                          filename=None,
                          verbose=False):
    container_path = config.valid_dir
    if not filename:
        filename = config.y_valid_path
    return get_y_from_path(
        classes=classes,
        container_path=container_path,
        shuffle=shuffle,
        random_state=random_state,
        save=save,
        filename=filename,
        verbose=verbose)


def get_y_from_path_test(classes=None,
                         shuffle=False,
                         random_state=0,
                         save=False,
                         filename=None,
                         verbose=False):
    container_path = config.test_dir
    if not filename:
        filename = config.y_test_path
    return get_y_from_path(
        classes=classes,
        container_path=container_path,
        shuffle=shuffle,
        random_state=random_state,
        save=save,
        filename=filename,
        verbose=verbose)
