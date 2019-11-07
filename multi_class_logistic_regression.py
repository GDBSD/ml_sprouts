# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix


class MulticlassLogisticRegressionDemo:
    """A simple multiclass regression model.
    O'Reilly E-book page:
    https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_20_Chapter.xhtml
    """

    @staticmethod
    def make_prediction(data):
        """Train a linear regression model and evaluate is quality.

        :param data: tuple - a tuple containing the data and the targets.
        :return:
        """
        pass

        # Separate features and target
        X = data[0]
        y = data[1]

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

        # Create the model
        logistic_reg = LogisticRegression(solver='lbfgs', multi_class="ovr")

        # Fit the model on the training set
        logistic_reg.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = logistic_reg.predict(X_test)

        # Evaluate the model performance using accuracy metric
        accuracy = accuracy_score(y_test, predictions)
        print('Accuracy {:.2f}'.format(accuracy))

        confusion_matrix = multilabel_confusion_matrix(y_test, predictions)
        print('Confusion Matrix:')
        print(confusion_matrix)


if __name__ == "__main__":

    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    sample_data = datasets.load_iris(return_X_y=True)

    predictor = MulticlassLogisticRegressionDemo()
    predictor.make_prediction(sample_data)
