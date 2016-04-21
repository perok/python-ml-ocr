import numpy as np
import logging
from skimage.color import rgb2grey

logger = logging.getLogger(__name__)


def sliding_window(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def perform_ocr(classifier, image, image_size=(20, 20), min_percentage=0.5):
    """

    :param classifier:
    :param image:
    :param image_size: (width, height)
    :param min_percentage: default is 0.5
    :return: yield ((x, y), (width, height), predicted char, percentage)
    """
    logger.info('Starting OCR')

    for (x, y, window) in sliding_window(image, step_size=5, window_size=image_size):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != image_size[0] or window.shape[1] != image_size[1]:
            continue

        img = np.reshape(window, (-1, image_size[0] * image_size[1]))
        #img = img.reshape(1, -1)
        #print(window)

        result = classifier.predict_proba(img)  # img[np.newaxis, ...]
        result = max(enumerate(result[0]), key=lambda _x: _x[1])
        if result[1] > min_percentage:
            yield((x, y), image_size, chr(result[0] + 97), result[1])

    logger.info('Finished with OCR')

