from pathlib import Path
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

from deepposekit.models import load_model
from deepposekit.io import DataGenerator, ImageGenerator
from deepposekit.io import DataGenerator, VideoReader, VideoWriter
from deepposekit.io.utils import merge_new_images
from os.path import expanduser

import tqdm
import time

from scipy.signal import find_peaks

from os.path import expanduser

def convert_resize(root_path, filepath):

    print("Let's start!")
    path = os.path.join(root_path, '%s_fish_roi.avi' % filepath)
    cap = cv2.VideoCapture(path)
    print("yeah, change the size of the movie")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    path2 = os.path.join(root_path, '%s_fish_roi_resized.avi' % filepath)
    out = cv2.VideoWriter(path2, fourcc, 5, (96, 96))

    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame, (96, 96), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(b)
        else:
            print("oh no")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done done!")


def initialize_annotation(root_path, filepath):
    path = os.path.join(root_path, 'my_best_model.h5')
    model = load_model(path)
    path2 = os.path.join(root_path, 'example_annotation_set.h5')
    data_generator = DataGenerator(path2, mode='unannotated')
    image_generator = ImageGenerator(data_generator)

    predictions = model.predict(image_generator, verbose=1)

    print(predictions.shape)

    data_generator[:] = predictions

    image, keypoints = data_generator[0]

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
    # path3 = os.path.join(root_path, "figure6_%s.png" % filepath)
    # plt.savefig(path3)
    # plt.show()

    path = os.path.join(root_path, 'my_best_model.h5')
    model = load_model(path)

    path2 = os.path.join(root_path, '%s_fish_roi_resized.avi' % filepath)
    reader = VideoReader(path2, batch_size=10, gray=True)
    frames = reader[0]
    print(frames.shape)
    reader.close()
    fish_name = filepath

    # plt.imshow(frames[0,...,0], cmap='gray')
    # plt.savefig('figure_15_%s.png' % filepath)
    # plt.show()

    reader = VideoReader(path2, batch_size=50, gray=True)
    predictions = model.predict(reader, verbose=1)
    reader.close()

    path3 = os.path.join(root_path, filepath)
    np.save(path3 + '_predictions.npy', predictions)

    x, y, confidence = np.split(predictions, 3, -1)
    path4 = os.path.join(root_path, 'example_annotation_set.h5')
    data_generator = DataGenerator(path4)

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
    plt.savefig(path3 + "_figure7.png")
    # plt.show()

    confidence_diff = np.abs(np.diff(confidence.mean(-1).mean(-1)))

    confidence_outlier_peaks = find_peaks(confidence_diff, height=0.1)[0]

    time_diff = np.diff(predictions[..., :2], axis=0)
    time_diff = np.abs(time_diff.reshape(time_diff.shape[0], -1))
    time_diff = time_diff.mean(-1)

    time_diff_outlier_peaks = find_peaks(time_diff, height=10)[0]

    outlier_index = np.concatenate((confidence_outlier_peaks, time_diff_outlier_peaks))
    outlier_index = np.unique(outlier_index) # make sure there are no repeats

    reader = VideoReader(path2, batch_size=1, gray=True)

    outlier_images = []
    outlier_keypoints = []
    for idx in outlier_index:
        outlier_images.append(reader[idx])
        outlier_keypoints.append(predictions[idx])

    outlier_images = np.concatenate(outlier_images)
    outlier_keypoints = np.stack(outlier_keypoints)

    reader.close()

    data_generator = DataGenerator(path4)

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
        plt.savefig(path3 + '_figure_12_%s.png' % filepath)
        plt.show()

    merge_new_images(
        datapath=path4,
        merged_datapath=path3 + '_annotation_data_release_merged.h5',
        images=outlier_images,
        keypoints=outlier_keypoints,
        # overwrite=True # This overwrites the merged dataset if it already exists
    )

    merged_generator = DataGenerator(path3 + '_annotation_data_release_merged.h5', mode="unannotated")

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
    plt.savefig(path3 + '_figure_13.png')
    plt.show()

    plt.imshow(frame[...,::-1])
    plt.savefig(path3 + '_figure_14.png')

    plt.show()

if __name__ == '__main__':
    movie_path = Path(os.environ["MOVIE_PATH"]).resolve()

    # this comes from the command line
    filepath = Path(sys.argv[1]).resolve()
    root_path = filepath.parent
    movie_name = filepath.stem[-21]
    print(movie_name)

    # if (root_path / f"{movie_name}_fish_roi_resized.avi").exists():
    #     #     print("Stack already converted. Skipping.")
    #     #     sys.exit()

    # convert_resize(root_path=root_path, filepath=movie_name)
    initialize_annotation(root_path=root_path, filepath=movie_name)


