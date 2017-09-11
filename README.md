# transfer_learning

Overview
--------
transfer_learning is a keras-based transfer learning module for arbitrary end-to-end image classification. The module supports transfer learning that is based on any of the following pre-trained models

- [Xception](https://keras.io/applications/#xception)
- [VGG16](https://keras.io/applications/#vgg16)
- [VGG19](https://keras.io/applications/#vgg19)
- [ResNet50](https://keras.io/applications/#resnet50)
- [InceptionV3](https://keras.io/applications/#inceptionv3)
- [MobileNet](https://keras.io/applications/#mobilenet)

Setup
-----
Download all the Python files of the module into the project folder. In order for the module to correctly find the data, the following folder structure is required to store the data

```
project_dir/
    data/
        train/
            category_1
                image_11
                image_12
                ...
            category_2
                image_21
                image_22
                ...
            ...
            category_N
                image_N1
                image_N2
                ...
        valid/
            ...
        test/
            ...
```

Usuage
------
```
python transfer_learning.py -m resnet50
```
