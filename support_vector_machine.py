# -*- coding: utf-8 -*-

# import packages
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""A simple SVM model.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_22_Chapter.xhtml
"""


class SupportVectorMachineDemo:
    """In this code example, we will build an SVM classification model to predict the three
    species of flowers from the Iris dataset. In the real world, it is difficult to find data
    points that are precisely linearly separable and for which exists a large margin hyperplane.
    The goal of the support vector classifier is to find a hyperplane that nearly
    discriminates between the two classes. This technique is also called a soft margin."""

    @staticmethod
    def make_prediction(data):
        """In this code example, we will build an SVM classification model to predict the three
        species of flowers from the Iris dataset. In the real world, it is difficult to find data
        points that are precisely linearly separable and for which exists a large margin hyperplane.
        The goal of the support vector classifier is to find a hyperplane that nearly
        discriminates between the two classes. This technique is also called a soft margin.

        :param data: tuple - a tuple containing the data and the targets.
        :return:
        """

        # Separate features and target
        X = data[0]
        y = data[1]

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

        # Create the model
        svc_model = SVC(gamma='scale')

        # Fit the model on the training set
        svc_model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = svc_model.predict(X_test)

        # Evaluate the model performance using accuracy metric
        accuracy = accuracy_score(y_test, predictions)
        print('Accuracy {:.2f}'.format(accuracy))


if __name__ == "__main__":

    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    sample_data = datasets.load_iris(return_X_y=True)

    predictor = SupportVectorMachineDemo()
    predictor.make_prediction(sample_data)