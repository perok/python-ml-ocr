import gzip
import logging
import os
import pickle

import numpy as np
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from data import get_label_images, data_augmentation

logger = logging.getLogger(__name__)


class KNN(object):
    def __init__(self, restore=True, restore_path='/tmp/tfmodels/knn.pklz'):
        self.__name__ = __name__
        self.classifier = None
        self.restore_path = restore_path

        if os.path.isfile(restore_path) and restore:
            with gzip.open(restore_path, 'rb') as f:
                classifier = pickle.load(f)
            logger.info('Restored classifier from file')
            self.classifier = classifier
        else:
            self.create_classifier()

    def create_classifier(self):
        logger.info('Starting to build {0} classifier'.format(__name__))
        images, labels = get_label_images('/var/dataset/chars74k-lite')
        # TODO test ut Otsu method https://derekjanni.github.io/pyocr/ipynb.html
        n_classes = len(set(labels))
        logger.info('Classifying {0} labels: {1} '.format(n_classes, set(labels)))
        logger.info('Found {0} images'.format(len(images)))

        # Split dataset into training and test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42)

        # Preprocess the training set
        X_train, y_train = data_augmentation(X_train, y_train, config={
            'noise': [{'mode': 'gaussian'}, {'mode': 's&p'}, {'mode': 'poisson'}],
            'roll': [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        })
        X_train = np.array([np.reshape(image, 20 * 20) for image in X_train])
        X_test = np.array([np.reshape(image, 20 * 20) for image in X_test])

        # Shuffle the augmented set
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, test_size=0, random_state=42)

        #logger.info('Example image\n{0}'.format(X_train[0]))

        # Preprocessing components
        pca = PCA(n_components=50) #RandomizedPCA
        scaler = StandardScaler()

        logger.info('Training set: {0}'.format(len(X_train)))
        logger.info('Validation set: {0}'.format(len(X_test)))

        classifier = KNeighborsClassifier()

        pipeline = Pipeline([
            ("pca", pca),
            ("mms", scaler),  # MinMaxScaler()
            ("knn", classifier)])

        pipeline.fit(X_train, y_train)
        logger.info('Pipeline:\n{0}'.format(pipeline))

        #lol = pca.transform(X_train)
        #logger.info('Example image\n{0}'.format(lol[0]))
        #logger.info('Example image\n{0}'.format(scaler.transform(lol)[0]))

        #logger.info('Explained variance by components: {0:.2f}%'.format(np.sum(pca.explained_variance_ratio_)))

        y_test_pred = pipeline.predict(X_test)
        score = metrics.accuracy_score(y_test, y_test_pred)
        logger.info('Accuracy: {0:f}'.format(score))
        logger.info('Classification report\n{0}'.format(metrics.classification_report(y_test, y_test_pred)))
        #logger.info('Confusion matrix:\n{0}'.format(confusion_matrix(y_test, y_test_pred)))
        #logger.info('Cross validation on train set: {0}'.format(cross_validation.cross_val_score(classifier, X_train, y_train, cv=5)))

        with gzip.open(self.restore_path, 'wb') as f:
            pickle.dump(classifier, f)

        logger.info('Classifier saved')
        self.classifier = pipeline
