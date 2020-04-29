from deepposekit.io import TrainingGenerator
from deepposekit.models import StackedDenseNet
from deepposekit.models import load_model

from deepposekit.io import DataGenerator, VideoReader, VideoWriter
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia
import pylab as plt
import numpy as np
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepposekit.models import (StackedDenseNet,
                                DeepLabCut,
                                StackedHourglass,
                                LEAP)
import glob

from deepposekit.callbacks import Logger, ModelCheckpoint
# glob.glob(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/*annotation*.h5')
data_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5')
# data_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5')

augmenter = []

augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right

sometimes = []
sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                            shear=(-8, 8),
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode=ia.ALL)
                 )
sometimes.append(iaa.Affine(scale=(0.8, 1.2),
                            mode=ia.ALL,
                            order=ia.ALL,
                            cval=ia.ALL)
                 )
augmenter.append(iaa.Sometimes(0.75, sometimes))
augmenter.append(iaa.Affine(rotate=(-180, 180),
                            mode=ia.ALL,
                            order=ia.ALL,
                            cval=ia.ALL)
                 )
augmenter = iaa.Sequential(augmenter)


for k in range(10):
    image, keypoints = data_generator[5]
    image, keypoints = augmenter(images=image, keypoints=keypoints)

    plt.figure(figsize=(5,5))
    image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )
    plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)
    plt.savefig(f"/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/{k}.png")


train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=0,
                                    augmenter=augmenter,
                                    sigma=5,
                                    validation_split=0.1,
                                    use_graph=True,
                                    random_seed=1,
                                    graph_scale=1)
train_generator.get_config()

# model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True, n_transitions=1)

# model = DeepLabCut(train_generator, backbone="resnet50")
# model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=0.35) # Increase alpha to improve accuracy
# model = DeepLabCut(train_generator, backbone="densenet121")

model = LEAP(train_generator)
#model = StackedHourglass(train_generator)

model.get_config()
"""
data_size = (10000,) + data_generator.image_shape
x = np.random.randint(0, 255, data_size, dtype="uint8")
y = model.predict(x[:100], batch_size=100) # make sure the model is in GPU memory
t0 = time.time()
y = model.predict(x, batch_size=100, verbose=1)
t1 = time.time()
print(x.shape[0] / (t1 - t0))
"""

logger = Logger(validation_batch_size=10)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)

model_checkpoint = ModelCheckpoint('/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_best_model.h5',
    monitor="val_loss",
    # monitor="loss" # use if validation_split=0
    verbose=1,
    save_best_only=True,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    # monitor="loss" # use if validation_split=0
    min_delta=0.001,
    patience=100,

verbose=1
)

callbacks = [early_stop, reduce_lr, model_checkpoint, logger]

model.fit(
    batch_size=16,
    validation_batch_size=10,
    callbacks=callbacks,
    #epochs=1000, # Increase the number of epochs to train the model longer
    epochs=200,
    n_workers=8,
    steps_per_epoch=None,
)