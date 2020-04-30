import numpy as np

import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import load_sample_image
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    #%tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"


if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")


# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#%matplotlib inline


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "project_cnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
    
def crop(images):
    return images[150:220, 130:250]

if __name__ == '__main__':
    
    folder1 = 'project_spect'
    ex1_path = os.path.join(PROJECT_ROOT_DIR, "data", folder1, '00907299.png' )
    ex2_path = os.path.join(PROJECT_ROOT_DIR, "data", folder1, '00907479.png')
    
    # Load sample images
    ex1 = load_sample_image(ex1_path) / 255
    ex2 = load_sample_image(ex2_path) / 255
    images = np.array([ex1, ex2])
    batch_size, height, width, channels = images.shape
    
    # Create 2 filters
    filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1  # vertical line
    filters[3, :, :, 1] = 1  # horizontal line
    
    outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
    
    plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
    plt.axis("off") # Not shown in the book
    plt.show()
    
    for image_index in (0, 1):
        for feature_map_index in (0, 1):
            plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
            plot_image(outputs[image_index, :, :, feature_map_index])

    plt.show()

    plot_image(crop(images[0, :, :, 0]))
    save_fig("ex1_original", tight_layout=False)
    plt.show()
    
    for feature_map_index, filename in enumerate(["ex1_vertical", "ex1_horizontal"]):
        plot_image(crop(outputs[0, :, :, feature_map_index]))
        save_fig(filename, tight_layout=False)
        plt.show()

    plot_image(filters[:, :, 0, 0])
    plt.show()
    plot_image(filters[:, :, 0, 1])
    plt.show()

    conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding="SAME", activation="relu")

    plot_image(crop(outputs[0, :, :, 0]))
    plt.show()

