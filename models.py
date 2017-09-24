import os
import numpy as np
import config
import utils
import data
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class TransferModel(object):
    def __init__(self,
                 base_model=None,
                 fc_layer_size=2048,
                 classes=None,
                 freeze_layers_num=None):
        if not base_model:
            base_model = config.model
        assert utils.is_keras_pretrained_model(base_model)
        self.base_model = base_model
        self.input_shape = utils.get_input_shape(self.base_model)
        self.fc_layer_size = fc_layer_size
        if classes is None:
            classes = config.classes
        self.classes = classes
        self.output_dim = len(classes)
        self.image_size = config.target_size_dict[base_model]
        if freeze_layers_num is None:
            freeze_layers_num = 80
        self.freeze_layers_num = freeze_layers_num
        self.model_weights_path = config.get_transfer_model_weights_path(
            base_model)
        self.model_path = config.get_transfer_model_path(base_model)
        self.preprocess_fun = data.preprocess_input_wrapper(self.base_model)
        self._create()

    def _create(self):
        model = utils.get_pretrained_model(
            self.base_model,
            include_top=False,
            input_shape=self.input_shape)
        interm = model.output
        interm = Flatten()(interm)
        interm = Dropout(0.5)(interm)
        interm = Dense(self.fc_layer_size, activation='relu')(interm)
        interm = Dropout(0.5)(interm)
        output = Dense(len(self.classes), activation='softmax')(interm)
        self.model = Model(inputs=model.input, outputs=output)

    def fit(self,
            x=None,
            y=None,
            batch_size=32,
            epochs=250,
            verbose=1,
            callbacks=None,
            validation_data=None,
            lr=1e-4,
            **kwargs):
        self.freeze_top_layers(self.model, self.freeze_layers_num)
        if x is None:
            x_train_path = config.get_x_train_path(self.base_model)
            x = utils.load_h5file(x_train_path)
        if y is None:
            y_train_path = config.y_train_path
            y = utils.load_h5file(y_train_path)
        if callbacks is None:
            callbacks = self.get_callbacks(
                self.model_weights_path,
                patience=30)
        if validation_data is None:
            x_valid_path = config.get_x_valid_path(self.base_model)
            x_valid = utils.load_h5file(x_valid_path)
            y_valid_path = config.y_valid_path
            y_valid = utils.load_h5file(y_valid_path)
            validation_data = (x_valid, y_valid)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=lr, momentum=0.9),
            metrics=['accuracy'])
        self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            **kwargs)

    def fit_generator(self,
                      generator=None,
                      steps_per_epoch=None,
                      epochs=250,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      lr=1e-4,
                      batch_size=32,
                      source='path',
                      **kwargs):
        self.freeze_top_layers(self.model, self.freeze_layers_num)
        assert source in {'path', 'tensor'}
        if generator is None:
            datagen_train = ImageDataGenerator(
                preprocessing_function=self.preprocess_fun,
                rotation_range=30.,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
            x_train_path = config.get_x_train_path(self.base_model)
            y_train_path = config.y_train_path
            if (source == 'tensor' and
                    os.path.exists(x_train_path) and
                    os.path.exists(y_train_path)):
                x_train = utils.load_h5file(x_train_path)
                y_train = utils.load_h5file(y_train_path)
                generator = datagen_train.flow(x_train, y_train, batch_size)
                n_train = len(x_train)
            else:
                generator = datagen_train.flow_from_directory(
                    config.train_dir,
                    target_size=self.image_size,
                    batch_size=batch_size)
                n_train = len(utils.images_under_subdirs(
                    config.train_dir,
                    subdirs=self.classes))
            if not steps_per_epoch:
                steps_per_epoch = utils.ceildiv(n_train, batch_size)
        if steps_per_epoch is None:
            steps_per_epoch = 500
        if not callbacks:
            callbacks = self.get_callbacks(
                self.model_weights_path,
                patience=50)
        if validation_data is None:
            datagen_valid = ImageDataGenerator(
                preprocessing_function=self.preprocess_fun)
            x_valid_path = config.get_x_valid_path(self.base_model)
            y_valid_path = config.y_valid_path
            if (source == 'tensor' and
                    os.path.exists(x_valid_path) and
                    os.path.exists(y_valid_path)):
                x_valid = utils.load_h5file(x_valid_path)
                y_valid = utils.load_h5file(y_valid_path)
                validation_data = datagen_valid.flow(
                    x_valid, y_valid, batch_size)
                n_valid = len(x_valid)
            else:
                validation_data = datagen_valid.flow_from_directory(
                    config.valid_dir,
                    target_size=self.image_size,
                    batch_size=batch_size)
                n_valid = len(utils.images_under_subdirs(
                    config.valid_dir,
                    subdirs=self.classes))
            if not validation_steps:
                validation_steps = utils.ceildiv(n_valid, batch_size)
        if validation_steps is None:
            validation_steps = 100
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=lr, momentum=0.9),
            metrics=['accuracy'])
        self.model.fit_generator(
            generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_from_path(self, image_path):
        tensor = data.path_to_tensor(image_path, target_size=self.target_size)
        tensor = self.preprocess_fun(tensor)
        pred = self.predict(tensor)
        return pred

    def predict_from_url(self, url):
        image_path = utils.url2file(url)
        pred = self.predict_from_path(image_path)
        return pred

    def save_model(self):
        self.model.save(self.model_path)

    def load_weights(self, weights_path=None):
        if not weights_path:
            weights_path = self.weights_top_model_path
        self.model.load_weights(weights_path)

    def load_weights_from_top_model(self, top_model_weights_path=None):
        if top_model_weights_path is None:
            top_model_weights_path = config.get_top_model_weights_path(
                self.base_model)
        pretrained_model = utils.get_pretrained_model(
            self.base_model,
            include_top=False,
            input_shape=self.input_shape)
        top_model = TopModel(
            base_model=self.base_model,
            fc_layer_size=self.fc_layer_size,
            classes=self.classes)
        top_model.load_weights(top_model_weights_path)
        model = Model(
            inputs=pretrained_model.input,
            outputs=top_model.model(pretrained_model.output))
        self.model = model

    @staticmethod
    def make_model_layers_nontrainable(model):
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def freeze_top_layers(model, freeze_layers_num):
        num_layers = len(model.layers)
        if freeze_layers_num >= 0 and freeze_layers_num <= num_layers:
            for layer in model.layers[:freeze_layers_num]:
                layer.trainable = False
            for layer in model.layers[freeze_layers_num:]:
                layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path,
                      monitor='val_loss',
                      patience=40,
                      patience_lr=20):
        early_stopping = EarlyStopping(
            verbose=1,
            patience=patience,
            monitor=monitor)
        model_checkpoint = ModelCheckpoint(
            weights_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            monitor=monitor)
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor, factor=0.2, patience=patience_lr, min_lr=1e-8)
        return [early_stopping, model_checkpoint, reduce_lr]


class TopModel(object):
    """
    An image classification model that is built on bottleneck features of a
    pre-trained model chosen from
        - 'inception_v3',
        - 'mobilenet',
        - 'resnet50',
        - 'resnet101',
        - 'resnet152',
        - 'vgg16',
        - 'vgg19',
        - 'xception'
    """
    def __init__(self,
                 base_model=None,
                 fc_layer_size=2048,
                 classes=None):
        self.fc_layer_size = fc_layer_size
        if not base_model:
            base_model = config.model
        assert utils.is_keras_pretrained_model(base_model)
        self.base_model = base_model
        if classes is None:
            classes = config.classes
        self.classes = np.array(classes)
        self.output_dim = len(classes)
        self.image_size = config.target_size_dict[base_model]
        self.model_weights_path = config.get_top_model_weights_path(base_model)
        self.model_path = config.get_top_model_path(base_model)
        self.preprocess_fun = data.preprocess_input_wrapper(self.base_model)
        self._create()

    def _create(self):
        pretrained_model = utils.get_pretrained_model(
            self.base_model,
            include_top=False,
            input_shape=utils.get_input_shape(self.base_model))
        self.pretrained_model = pretrained_model
        input_shape = [int(ele) for ele in pretrained_model.output.shape[1:]]
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(self.fc_layer_size, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_dim, activation='softmax'))
        self.model = model

    def fit(self,
            x=None,
            y=None,
            batch_size=32,
            epochs=250,
            verbose=1,
            callbacks=None,
            validation_data=None,
            lr=1e-3,
            **kwargs):
        if x is None:
            x_train_path = config.get_bf_train_path(self.base_model)
            x = utils.load_h5file(x_train_path)
        if y is None:
            y_train_path = config.y_train_path
            y = utils.load_h5file(y_train_path)
        if callbacks is None:
            callbacks = self.get_callbacks(
                self.model_weights_path,
                patience=30)
        if validation_data is None:
            x_valid_path = config.get_bf_valid_path(self.base_model)
            x_valid = utils.load_h5file(x_valid_path)
            y_valid_path = config.y_valid_path
            y_valid = utils.load_h5file(y_valid_path)
            validation_data = (x_valid, y_valid)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=lr, momentum=0.9),
            metrics=['accuracy'])
        self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def save_model(self):
        self.model.save(self.model_path)

    def load_weights(self, weights_path=None):
        if not weights_path:
            weights_path = self.model_weights_path
        self.model.load_weights(weights_path)

    def predict_from_path(self, image_path):
        tensor = data.path_to_tensor(image_path, target_size=self.target_size)
        tensor = self.preprocess_fun(tensor)
        bf = self.pretrained_model.predict(tensor)
        pred = self.predict(bf)
        return pred

    def predict_from_url(self, url):
        image_path = utils.url2file(url)
        pred = self.predict_from_path(image_path)
        return pred

    @staticmethod
    def get_callbacks(weights_path,
                      monitor='val_loss',
                      patience=30,
                      patience_lr=10):
        early_stopping = EarlyStopping(
            verbose=1,
            patience=patience,
            monitor=monitor)
        model_checkpoint = ModelCheckpoint(
            weights_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            monitor=monitor)
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor, factor=0.2, patience=patience_lr, min_lr=1e-8)
        return [early_stopping, model_checkpoint, reduce_lr]
