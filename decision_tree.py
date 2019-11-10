# -*- coding: utf-8 -*-

from math import sqrt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

"""A simple decision tree model.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_23_Chapter.xhtml
"""

# Make the models deterministic
RANDOM_SEED = 42

class DecisionTreeDemo:
    """In this code example, we will build a decision tree classification model and a decision tree
    regression.

    One of the significant advantages of this class of models is that they perform well on linear and
    non-linear datasets. Moreover, they implicitly take care of feature selection and work well with
    high-dimensional datasets. On the flip side, these models can very easily overfit the dataset and
    fail to generalize to new examples. This downside is mitigated by aggregating a large number of
    decision trees in techniques like Random forests and boosting ensemble algorithms. There are demo
    examples of these two models here in misc_stuff.
    """

    def make_prediction(self, data, dataset):
        """Train a models and predict using the specified dataset.

        :param data: tuple - a tuple containing the data and the targets.
        :param dataset: string - the particular dataset we will train on.
        :return:
        """
        # Separate features and target
        X = data[0]
        y = data[1]

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

        # Create the model
        classifier = self.get_classifier(dataset)

        # Fit the model on the training set
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the model performance using accuracy metric
        if dataset == 'flowers':
            accuracy = accuracy_score(y_test, predictions)
            print('Accuracy {:.2f}'.format(accuracy))
        else:
            rmse = sqrt(mean_squared_error(y_test, predictions))
            print('Root mean squared error {:.2f}'.format(rmse))

    @staticmethod
    def get_classifier(dataset_name):
        if dataset_name == 'flowers':
            return DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED)
        else:
            return DecisionTreeRegressor(max_depth=2, random_state=RANDOM_SEED)


if __name__ == "__main__":

    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    flower_data = datasets.load_iris(return_X_y=True)
    housing_data = datasets.load_boston(return_X_y=True)

    # Predict with the two models and the two datasets.
    predictor = DecisionTreeDemo()
    predictor.make_prediction(flower_data, 'flowers')
    predictor.make_prediction(housing_data, 'housing')
