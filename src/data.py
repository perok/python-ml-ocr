import logging
import os
import re
from glob import glob

import numpy as np
from skimage import io, img_as_float
from skimage.util import random_noise

logger = logging.getLogger(__name__)


def get_label_images(directory):
    # /var/dataset/chars74k-lite/g/g_110.jpg -> g
    char_matcher = re.compile('(\D)_\d+.jpg')

    image_files = glob(os.path.join(directory, '**/*.jpg'))

    # Get label and convert to zero indexing a..z
    # TODO 0..1 or -0.5..0.5
    labels = [ord(char_matcher.search(f).group(1)) - 97 for f in image_files] #np.array, dtype=np.uint32)
    #images = [img_as_float(io.imread(image).astype(np.uint8)) for image in image_files]
    images = [io.imread(image) for image in image_files]

    return images, labels


def data_augmentation(images, labels, config):
    dataset = list(zip(images, labels))
    extra_dataset = []

    if 'noise' in config:
        for noise in config['noise']:
            logger.info('Applying noise with: {0}'.format(noise))
            for image, label in dataset:
                extra_dataset.append((random_noise(image, **noise), label))

    if 'roll' in config:
        for roll in config['roll']:
            logger.info('Rolling dataset with: {0} axis={1}'.format(*roll))
            for image, label in dataset:
                extra_dataset.append((np.roll(image, *roll), label))

    dataset.extend(extra_dataset)
    images = np.array([image[0] for image in dataset])
    labels = np.array([label[1] for label in dataset], dtype=np.uint32)

    # images = images[..., np.newaxis]
    return images, labels

