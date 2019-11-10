# -*- coding: utf-8 -*-

from math import sqrt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Make the models deterministic
RANDOM_SEED = 42


class MulticlassLogisticRegressionDemo:
    """A simple multiclass regression model.
    O'Reilly E-book page:
    https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_20_Chapter.xhtml
    """

    def make_prediction(self, data, model_type):
        """Train a linear regression model and evaluate its quality.

        :param data: tuple - a tuple containing the data and the targets.
        :param model_type: string - type of model we want use, regression or ridge
        :return:
        """
        pass

        # Separate features and target
        X = data[0]
        y = data[1]

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

        # Create the model
        model = self.get_classifier(model_type)

        # Fit the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Evaluate the model performance.
        if model_type == 'logistic':
            accuracy = accuracy_score(y_test, predictions)
            print('Logistic regression accuracy {:.2f}'.format(accuracy))
        else:
            rmse = sqrt(mean_squared_error(y_test, predictions))
            print('Ridge regression root mean squared error {:.2f}'.format(rmse))

    @staticmethod
    def get_classifier(model_type):
        if model_type == 'flowers':
            return RidgeClassifier(random_state=RANDOM_SEED)
        else:
            return LogisticRegression(solver='lbfgs', multi_class="ovr", random_state=RANDOM_SEED)


if __name__ == "__main__":

    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    sample_data = datasets.load_iris(return_X_y=True)

    predictor = MulticlassLogisticRegressionDemo()
    predictor.make_prediction(sample_data, model_type='logistic')
    predictor.make_prediction(sample_data, model_type='ridge')
