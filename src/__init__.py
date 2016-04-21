import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from OCR import perform_ocr
from ConvNet import ConvNet
from KNN import KNN
from skimage import io, img_as_float
from os.path import basename
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


img1 = '/var/dataset/ocr/test.jpg'
img2 = '/var/dataset/ocr/a-s.jpg'
img3 = '/var/dataset/ocr/test2.png'


def create_ocr_fig(classifier, image_path):
    image = io.imread(image_path) #img_as_float(io.imread(image_path))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)

    for ((x, y), (width, height), cls, percentage) in perform_ocr(
            classifier, image,
            min_percentage=0.9):
        logger.info('({0}, {1}, {2}, {3}): {4} - {5:.2f}%'.format(x, y, width, height, cls, percentage))

        rect = mpatches.Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    return fig


def run_with(clfs, image_path):

   # fh = logging.FileHandler('/tmp/tflogs/{0}.log'.format(clfs.__name__))
   # logging.getLogger().addHandler(fh)

    figure = create_ocr_fig(clfs.classifier, image_path)
    figure.savefig('/var/dataset/{0}-{1}.jpg'.format(basename(image_path), clfs.__name__))

if __name__ == '__main__':
    cnn = ConvNet(restore=False)
    #knn = KNN(restore=False)

    run_with(cnn, img2)

