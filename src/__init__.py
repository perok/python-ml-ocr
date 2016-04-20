import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from OCR import perform_ocr
from ConvNet import get_conv_classifier
from KNN import get_knn_classifier
from skimage import io, img_as_float
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

img1 = '/var/dataset/ocr/test.jpg'
img2 = '/var/dataset/ocr/a-s.jpg'
img3 = '/var/dataset/ocr/test2.png'

def create_ocr_fig(classifier, image_path):
    image = img_as_float(io.imread(image_path))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)

    for ((x, y), (width, height), cls, percentage) in perform_ocr(
            classifier, image,
            min_percentage=0.9):
        logger.info('({0}, {1}, {2}, {3}): {4} - {5:.2f}%'.format(x, y, width, height, cls, percentage))

        rect = mpatches.Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    return fig


if __name__ == '__main__':
    #classifier_cnn = get_conv_classifier(restore=False)
    classifier_knn = get_knn_classifier(restore=False)

    figure = create_ocr_fig(classifier_knn, img2)
    figure.savefig('/var/dataset/result.jpg')
