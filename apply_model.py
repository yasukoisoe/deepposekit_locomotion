# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import glob
#
# from deepposekit.models import load_model
# from deepposekit.io import DataGenerator, ImageGenerator
#
# from os.path import expanduser
#
# try:
#     import google.colab
#     IN_COLAB = True
# except:
#     IN_COLAB = False
#
# HOME = expanduser("~") if not IN_COLAB else '.'
#
# models = sorted(glob.glob(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_best_model.h5'))
# # models
#
# annotations = sorted(glob.glob(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_annotation_set.h5'))
# print(annotations.shape)
#
# model = load_model(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_best_model.h5')
#
# data_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5', mode='unannotated')
# image_generator = ImageGenerator(data_generator)
#
# predictions = model.predict(image_generator, verbose=1)
#
# print(predictions.shape)
#
# data_generator[:] = predictions
#
# image, keypoints = data_generator[0]
#
# plt.figure(figsize=(5,5))
# image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
# cmap = None if image.shape[-1] is 3 else 'gray'
# plt.imshow(image, cmap=cmap, interpolation='none')
# for idx, jdx in enumerate(data_generator.graph):
#     if jdx > -1:
#         plt.plot(
#             [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
#             [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
#             'r-'
#         )
# plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)
# plt.savefig("figure6.png")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

from deepposekit.models import load_model
from deepposekit.io import DataGenerator, VideoReader, VideoWriter
from deepposekit.io.utils import merge_new_images

import tqdm
import time

from scipy.signal import find_peaks

from os.path import expanduser
# try:
#     import google.colab
#     IN_COLAB = True
# except:
#     IN_COLAB = False

# HOME = expanduser("~") if not IN_COLAB else '.'

models = sorted(glob.glob(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_best_model.h5'))

model = load_model(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_best_model.h5')

videos = sorted(glob.glob(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi'))

reader = VideoReader(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi', batch_size=10, gray=True)
frames = reader[0]
print(frames.shape)
reader.close()

plt.imshow(frames[0,...,0], cmap='gray')
plt.savefig('figure_15.png')
plt.show()

reader = VideoReader(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi', batch_size=50, gray=True)
predictions = model.predict(reader, verbose=1)
reader.close()

np.save(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/predictions.npy', predictions)

x, y, confidence = np.split(predictions, 3, -1)

data_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5')

image = frames[0]
keypoints = predictions[0]

plt.figure(figsize=(5,5))
image = image if image.shape[-1] is 3 else image[..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
for idx, jdx in enumerate(data_generator.graph):
    if jdx > -1:
        plt.plot(
            [keypoints[idx, 0], keypoints[jdx, 0]],
            [keypoints[idx, 1], keypoints[jdx, 1]],
            'r-'
        )
plt.scatter(keypoints[:, 0], keypoints[:, 1],
            c=np.arange(data_generator.keypoints_shape[0]),
            s=50, cmap=plt.cm.hsv, zorder=3)
plt.savefig("figure7.png")
plt.show()

confidence_diff = np.abs(np.diff(confidence.mean(-1).mean(-1)))

plt.figure(figsize=(15, 3))
plt.plot(confidence_diff)
plt.savefig("figure8.png")

plt.show()

confidence_outlier_peaks = find_peaks(confidence_diff, height=0.1)[0]

plt.figure(figsize=(15, 3))
plt.plot(confidence_diff)
plt.plot(confidence_outlier_peaks, confidence_diff[confidence_outlier_peaks], 'ro')
plt.savefig("figure9.png")

plt.show()

time_diff = np.diff(predictions[..., :2], axis=0)
time_diff = np.abs(time_diff.reshape(time_diff.shape[0], -1))
time_diff = time_diff.mean(-1)
print(time_diff.shape)

plt.figure(figsize=(15, 3))
plt.plot(time_diff)
plt.savefig("figure10.png")

plt.show()

time_diff_outlier_peaks = find_peaks(time_diff, height=10)[0]

plt.figure(figsize=(15, 3))
plt.plot(time_diff)
plt.plot(time_diff_outlier_peaks, time_diff[time_diff_outlier_peaks], 'ro')
plt.savefig("figure11.png")

plt.show()

outlier_index = np.concatenate((confidence_outlier_peaks, time_diff_outlier_peaks))
outlier_index = np.unique(outlier_index) # make sure there are no repeats

reader = VideoReader(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi', batch_size=1, gray=True)

outlier_images = []
outlier_keypoints = []
for idx in outlier_index:
    outlier_images.append(reader[idx])
    outlier_keypoints.append(predictions[idx])

outlier_images = np.concatenate(outlier_images)
outlier_keypoints = np.stack(outlier_keypoints)

reader.close()

print(outlier_images.shape, outlier_keypoints.shape)

data_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5')

for idx in range(5):
    image = outlier_images[idx]
    keypoints = outlier_keypoints[idx]

    plt.figure(figsize=(5,5))
    image = image if image.shape[-1] is 3 else image[..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[idx, 0], keypoints[jdx, 0]],
                [keypoints[idx, 1], keypoints[jdx, 1]],
                'r-'
            )
    plt.scatter(keypoints[:, 0], keypoints[:, 1],
                c=np.arange(data_generator.keypoints_shape[0]),
                s=50, cmap=plt.cm.hsv, zorder=3)
    plt.savefig('figure_12_%s.png' % idx)
    plt.show()

merge_new_images(
    datapath=r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/my_annotation_set.h5',
    merged_datapath=r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/annotation_data_release_merged.h5',
    images=outlier_images,
    keypoints=outlier_keypoints,
    # overwrite=True # This overwrites the merged dataset if it already exists
)

merged_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/annotation_data_release_merged.h5', mode="unannotated")

image, keypoints = merged_generator[0]

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
plt.savefig('figure_13.png')
plt.show()

plt.imshow(frame[...,::-1])
plt.savefig('figure_14.png')

plt.show()