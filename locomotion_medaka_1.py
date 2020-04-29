import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
from deepposekit.annotate import KMeansSampler
import tqdm
import glob
import pandas as pd

from os.path import expanduser

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

HOME = expanduser("~") if not IN_COLAB else '.'
videos = glob.glob(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi')
reader = VideoReader(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi', gray=True) #deepposekit-data/datasets/fly
frame = reader[0] # read a frame
reader.close()
print(frame.shape)
plt.figure(figsize=(5,5))
plt.imshow(frame[0,...,0])
plt.savefig("fly_figure0.png")
plt.show()

reader = VideoReader(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/fish_roi_resized.avi', batch_size=100, gray=True)

randomly_sampled_frames = []
for idx in tqdm.tqdm(range(len(reader)-1)):
    batch = reader[idx]
    random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
    randomly_sampled_frames.append(random_sample)
reader.close()

randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
print(randomly_sampled_frames.shape)
kmeans = KMeansSampler(n_clusters=10, max_iter=1000, n_init=10, batch_size=100, verbose=True)
kmeans.fit(randomly_sampled_frames)
kmeans.plot_centers(n_rows=2)
plt.savefig("fly_figure02.png")
plt.show()

kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(randomly_sampled_frames, n_samples_per_label=10)
print(kmeans_sampled_frames.shape)

initialize_dataset(
    images=kmeans_sampled_frames,
    datapath=r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5',
    skeleton=r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/skeleton.csv',
    overwrite=True # This overwrites the existing datapath
)


data_generator = DataGenerator(r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5', mode="full")

image, keypoints = data_generator[0]

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
plt.savefig("fly_figure03.png")
plt.show()


from deepposekit import Annotator
from os.path import expanduser
import glob
HOME = expanduser("~")

app = Annotator(datapath=r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/example_annotation_set.h5',
                dataset='images',
                skeleton=r'/Users/yasukoisoe/fishfishfish/deepposekit_locomotion/skeleton.csv',
                shuffle_colors=False,
                text_scale=0.2)

app.run()