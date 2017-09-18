# TransferLearning

Overview
--------
`transfer_learning` is a [keras](https://keras.io/)-based transfer learning module for arbitrary end-to-end image classification. The module supports transfer learning that is based on any of the following pre-trained models

- [Xception](https://keras.io/applications/#xception)
- [VGG16](https://keras.io/applications/#vgg16)
- [VGG19](https://keras.io/applications/#vgg19)
- [ResNet50](https://keras.io/applications/#resnet50)
- [InceptionV3](https://keras.io/applications/#inceptionv3)
- [MobileNet](https://keras.io/applications/#mobilenet)

Transfer learning takes place through the following two steps

- Quick learning via a top model
    + A top model is a pre-trained model with last block of fully connected layers being re-trained to accommodate a specific classification task.
- Fine tuning of unfreezed layers
    + Fine tuning begins by initializing its weights to those of a top model trained in first step and re-trains those unfreezed layers, typically the final fully connected layer block + its previous adjacent convolutional layer block.
    
Please refer to Francois Chollet's excellent [tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for the details of a top model and fine tuning process.

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
```shell
python transfer_learning.py -m resnet50
```


Contact
-------
If you have any questions or encounter any bugs, please contact the author (Feiyang Niu, Feiyang.Niu@gilead.com)
