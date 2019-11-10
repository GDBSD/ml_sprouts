# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

"""Examples of various resampling approaches.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_24_Chapter.xhtml
"""

# Make the models deterministic
RANDOM_SEED = 42


class Resampler:
    """A set of techniques that involve selecting a subset of the available dataset, training on that data
    subset, and using the remainder of the data to evaluate the trained model."""

    def resample_data(self, data):
        """Apply and compare two resampling algorithms.

        :param data: tuple - a tuple containing the data and the targets.
        :return:
        """
        X = data[0]
        y = data[1]

        print()
        self.apply_kfold_cross_validation(X, y)
        print('=' * 50)
        self.apply_locv(X, y)

    @staticmethod
    def apply_kfold_cross_validation(X, y):
        """Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds.

        :param X: Numpy array - features
        :param y: Numpy array - labels
        :return:
        """
        print('Calculating KFold accuracy ... ')

        # Initialize KFold to shuffle the data before splitting
        kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        # Create the model
        knn_clf = KNeighborsClassifier(n_neighbors=3)

        # Fit the model using cross validation
        cv_result = cross_val_score(knn_clf, X, y, cv=kfold)

        # Evaluate the model performance using accuracy metric
        print('KFold Accuracy: {:.2f}'.format(cv_result.mean() * 100.0, cv_result.std() * 100.0))

    @staticmethod
    def apply_locv(X, y):
        """In leave one out cross validation (LOOCV) just one example is assigned to the test set,
        and the model is trained on the remainder of the dataset. This process is repeated for all
        the examples in the dataset. This process is repeated until all the examples in the dataset
        have been used for evaluating the model.

        :param X: Numpy array - features
        :param y: Numpy array - labels
        :return:
        """

        print('Calculating LOOCV accuracy ... ')

        # Initialize LOOCV.
        loocv = LeaveOneOut()

        # Create the model
        knn_clf = KNeighborsClassifier(n_neighbors=3)

        # Fit the model using cross validation.
        cv_result = cross_val_score(knn_clf, X, y, cv=loocv)

        # Evaluate the model performance using accuracy metric
        print('KFold Accuracy: {:.2f}'.format(cv_result.mean() * 100.0, cv_result.std() * 100.0))


if __name__ == "__main__":
    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    flower_data = datasets.load_iris(return_X_y=True)
    housing_data = datasets.load_boston(return_X_y=True)
    resampler = Resampler()
    resampler.resample_data(flower_data)
