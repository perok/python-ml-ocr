from data import get_label_images, data_augmentation
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.contrib import skflow
import logging
logger = logging.getLogger(__name__)


# https://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/

def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(X, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    X = tf.reshape(X, [-1, 20, 20, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])  # from 7*7 to 5*5
    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


def get_conv_classifier(restore=True, restore_path='/tmp/tfmodels/convmodel'):
    if restore:
        try:
            classifier = skflow.TensorFlowEstimator.restore(restore_path)

            logger.info('Restored classifier from file')
            return classifier
        except ValueError:
            logger.exception('No model')
    return create_classifier(restore_path)


def create_classifier(save_path):
    logger.info('Starting to build {0} classifier'.format(__name__))
    images, labels = get_label_images('/var/dataset/chars74k-lite')
    n_classes = len(set(labels))
    logger.info('Classifying {0} labels: {1} '.format(n_classes, set(labels)))
    logger.info('Found {0} images'.format(len(images)))

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        images, labels, test_size=0.2, random_state=42)

    X_train, y_train = data_augmentation(X_train, y_train, config={
        'noise': [{'mode': 'gaussian'}, {'mode': 'poisson'}, {'mode': 's&p'}],
        'roll': []  # [(1, 0), (-1, 0), (1, 1), (-1, 1)]

    })

    # Shuffle the augmented set
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=1, random_state=42)

    X_train = np.array([np.reshape(image, 20 * 20) for image in X_train])
    X_test = np.array([np.reshape(image, 20 * 20) for image in X_test])

    logger.info('Training set:\t{0}'.format(len(X_train)))
    logger.info('Validation set:\t{0}'.format(len(X_test)))

    #val_monitor = skflow.monitors.ValidationMonitor(X_val, y_val,
    #                                                early_stopping_rounds=200,
    #                                                n_classes=3,
    #                                                 print_steps=50)
    #
    # provide a validation monitor with early stopping rounds and validation set
    # classifier.fit(X_train, y_train, val_monitor)

    # Training and predicting
    classifier = skflow.TensorFlowEstimator(  #                         20000
        model_fn=conv_model, n_classes=n_classes, batch_size=100, steps=2000, learning_rate=0.01)
    classifier.fit(X_train, y_train, logdir='/tmp/tflogs/convnet')
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    logger.info('Accuracy: {0:f}'.format(score))
    logger.info('Classification report\n{0}'.format(metrics.classification_report(y_test, classifier.predict(X_test))))

    classifier.save(save_path)

    logger.info('Classifier saved')

    return classifier

