import os
import importlib
import config
import numpy as np
import h5py
from functools import partial
from sklearn.utils import check_random_state
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
from utils import images_under_dir, images_under_subdirs, remove_file


def preprocess_an_image(image, model='resnet50'):
    """
    Wrapper around `keras.applications.{model}.preprocess_input()`

    Parameters
    ----------
    image: a 3D numpy array
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'vgg16',
            'vgg19',
            'xception'

    Returns
    -------
    A 3D numpy array (preprocessed image)
    """
    assert model in [
        'inception_v3', 'mobilenet', 'resnet50',
        'vgg16', 'vgg19', 'xception'
    ]
    module = importlib.import_module('keras.applications.{}'.format(model))
    preprocess_input = module.preprocess_input
    image_list = np.expand_dims(image, axis=0)
    image_processed = preprocess_input(image_list)
    return image_processed[0]


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
            'vgg16',
            'vgg19',
            'xception'
    """
    return partial(preprocess_an_image, model=model)


def extract_bottleneck_features(model=None,
                                container_path=config.train_dir,
                                classes=None,
                                save=False,
                                filename=None,
                                verbose=True):
    """
    """
    if not model:
        model = config.model
    assert model in [
        'inception_v3', 'mobilenet', 'resnet50',
        'vgg16', 'vgg19', 'xception'
    ]
    imagepaths = images_under_subdirs(container_path, subdirs=classes)
    tensor_list = []
    target_size = config.target_size_dict[model]
    if verbose:
        print('Started: images -> tensors')
    for path in imagepaths:
        image = load_img(path, target_size=target_size)
        tensor = img_to_array(image)
        tensor = np.expand_dims(tensor, axis=0)
        tensor_list.append(tensor)
    preprocess_fun = preprocess_input_wrapper(model)
    tensors = np.vstack(tensor_list)
    tensors = preprocess_fun(tensors)
    if verbose:
        print('Finished: images -> tensors')
    input_shape = config.get_input_shape(model)
    pretrained_model = config.get_pretrained_model(
        model,
        include_top=False,
        input_shape=input_shape)
    bottleneck_features = pretrained_model.predict(
        tensors,
        verbose=1 if verbose else 0)
    if save:
        if not filename:
            filename = 'bottleneck_features_{}.h5'.format(config.model)
        filepath = os.path.join(config.precomputed_dir, filename)
        remove_file(filepath)
        if verbose:
            print('Started saving {}'.format(filename))
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('data', data=bottleneck_features)
        if verbose:
            print('Finished saving {}'.format(filename))
    else:
        return bottleneck_features


def extract_bottleneck_features_train(model=None,
                                      classes=None,
                                      save=False,
                                      filename=None,
                                      verbose=True):
    container_path = config.train_dir
    if not filename:
        filename = 'bottleneck_features_train_{}.h5'.format(config.model)
    return extract_bottleneck_features(
        model,
        container_path=container_path,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def extract_bottleneck_features_valid(model=None,
                                      classes=None,
                                      save=False,
                                      filename=None,
                                      verbose=True):
    container_path = config.valid_dir
    if not filename:
        filename = 'bottleneck_features_valid_{}.h5'.format(config.model)
    return extract_bottleneck_features(
        model,
        container_path=container_path,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def extract_bottleneck_features_test(model=None,
                                     classes=None,
                                     save=False,
                                     filename=None,
                                     verbose=True):
    container_path = config.test_dir
    if not filename:
        filename = 'bottleneck_features_test_{}.h5'.format(config.model)
    return extract_bottleneck_features(
        model,
        container_path=container_path,
        classes=classes,
        save=save,
        filename=filename,
        verbose=verbose)


def extract_classes_from_path(container_path,
                              classes=None,
                              shuffle=False,
                              random_state=0,
                              save=False,
                              filename=None,
                              verbose=True):
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
        num_images = len(images_under_dir(subf, examine_by='extension'))
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
        if not filename:
            filename = 'targets_{}.h5'.format(config.model)
        filepath = os.path.join(config.precomputed_dir, filename)
        remove_file(filepath)
        if verbose:
            print('Started saving {}'.format(filename))
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('data', data=targets_one_hot_encode)
        if verbose:
            print('Finished saving {}'.format(filename))
    else:
        return targets_one_hot_encode


def extract_classes_from_path_train(classes=None,
                                    shuffle=False,
                                    random_state=0,
                                    save=False,
                                    filename=None,
                                    verbose=True):
    container_path = config.train_dir
    if not filename:
        filename = 'targets_train_{}.h5'.format(config.model)
    return extract_classes_from_path(
        classes=classes,
        container_path=container_path,
        shuffle=shuffle,
        random_state=random_state,
        save=save,
        filename=filename,
        verbose=verbose)


def extract_classes_from_path_valid(classes=None,
                                    shuffle=False,
                                    random_state=0,
                                    save=False,
                                    filename=None,
                                    verbose=True):
    container_path = config.valid_dir
    if not filename:
        filename = 'targets_valid_{}.h5'.format(config.model)
    return extract_classes_from_path(
        classes=classes,
        container_path=container_path,
        shuffle=shuffle,
        random_state=random_state,
        save=save,
        filename=filename,
        verbose=verbose)


def extract_classes_from_path_test(classes=None,
                                   shuffle=False,
                                   random_state=0,
                                   save=False,
                                   filename=None,
                                   verbose=True):
    container_path = config.test_dir
    if not filename:
        filename = 'targets_test_{}.h5'.format(config.model)
    return extract_classes_from_path(
        classes=classes,
        container_path=container_path,
        shuffle=shuffle,
        random_state=random_state,
        save=save,
        filename=filename,
        verbose=verbose)
