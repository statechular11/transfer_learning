import os
import config
import utils
import data
import models
import argparse


def transfer_learning(base_model=None,
                      fc_layer_size=2048,
                      freeze_layers_num=None,
                      classes=None,
                      epochs_top_model=250,
                      epochs_fine_tuned=250,
                      lr_top_model=1e-3,
                      lr_fine_tuned=1e-4,
                      project_path=None):
    if project_path is None:
        project_path = config.abspath
    config.trained_dir = os.path.join(project_path, 'trained')
    config.precomputed_dir = os.path.join(project_path, 'precomputed')
    utils.create_dir(config.trained_dir)
    utils.create_dir(config.precomputed_dir)
    config.get_top_model_weights_path(base_model)
    config.get_fine_tuned_weights_path(base_model)
    config.get_top_model_path(base_model)
    config.get_fine_tuned_path(base_model)
    if base_model is None:
        base_model = config.model
    assert base_model in [
        'inception_v3', 'mobilenet', 'resnet50',
        'vgg16', 'vgg19', 'xception'
    ]
    if classes is not None:
        classes = config.classes
    print('Started extracting bottleneck features for train data')
    x_train = data.extract_bottleneck_features_train(
        model=base_model,
        classes=classes,
        save=False,
        verbose=True)
    print('Finished extracting bottleneck features for train data')
    y_train = data.extract_classes_from_path_train(
        classes=classes,
        shuffle=False,
        save=False,
        verbose=True)
    print('Started extracting bottleneck features for valid data')
    x_valid = data.extract_bottleneck_features_valid(
        model=base_model,
        classes=classes,
        save=False,
        verbose=True)
    print('Finished extracting bottleneck features for valid data')
    y_valid = data.extract_classes_from_path_valid(
        classes=classes,
        shuffle=False,
        save=False,
        verbose=True)
    top_model = models.TopModel(
        base_model=base_model,
        fc_layer_size=fc_layer_size)
    top_model.fit(
        x_train,
        y_train,
        epochs=epochs_top_model,
        validation_data=(x_valid, y_valid),
        lr=lr_top_model)
    transfer_model = models.TransferModel(
        base_model=base_model,
        fc_layer_size=fc_layer_size)
    transfer_model.load_weights_from_top_model()
    transfer_model.fit_generator(epochs=epochs_fine_tuned, lr=lr_fine_tuned)
    return transfer_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--base_model',
                        type=str,
                        required=False,
                        help=('Base model architecture'),
                        default='resnet50',
                        choices=['inception_v3', 'mobilenet', 'resnet50',
                                 'vgg16', 'vgg19', 'xception'])
    parser.add_argument('--fc_layer_size',
                        type=int,
                        required=False,
                        help='Size of fully connected layer before output',
                        default=2048)
    parser.add_argument('--freeze_layers_num',
                        type=int,
                        required=False,
                        help=('Will freeze the first N layers and ',
                              'unfreeze the rest'))
    parser.add_argument('--epochs_top_model',
                        type=int,
                        required=False,
                        help=('Number of epochs for training top model'))
    parser.add_argument('--epochs_fine_tuned',
                        type=int,
                        required=False,
                        help=('Number of epochs for fine tuning'))
    parser.add_argument('--lr_top_model',
                        type=float,
                        required=False,
                        help=('Learning rate for training top model'),
                        default=0.001)
    parser.add_argument('--lr_fine_tuned',
                        type=float,
                        required=False,
                        help=('Learning rate for fine tuning'),
                        default=0.0001)
    parser.add_argument('-p', '--project_path',
                        action='store',
                        required=False,
                        help='Path of the project folder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    transfer_learning(base_model=args.base_model,
                      fc_layer_size=args.fc_layer_size,
                      freeze_layers_num=args.freeze_layers_num,
                      classes=None,
                      epochs_top_model=args.epochs_top_model,
                      epochs_fine_tuned=args.epochs_fine_tuned,
                      lr_top_model=args.lr_top_model,
                      lr_fine_tuned=args.lr_fine_tuned,
                      project_path=args.project_path)
