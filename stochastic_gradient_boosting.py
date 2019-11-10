# -*- coding: utf-8 -*-

from math import sqrt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

"""A simple Stochastic Gradient Boosting (SGB) model.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_23_Chapter.xhtml
"""

# Make the models deterministic
RANDOM_SEED = 42


class StochasticGradientBoostingDemo:
    """Stochastic Gradient Boosting (SGB) Boosting involves growing trees in succession using knowledge
    from the residuals of the previously grown tree. In this case, each successive tree works to improve
    the model of the previous tree by boosting the areas in which the previous tree did not perform so
    well without affecting the areas of high performance. By doing this, we iteratively create a model
    that reduces the residual variance when generalizing to test examples.
    """
    def make_prediction(self, data, dataset):

        # Separate features and target
        X = data[0]
        y = data[1]

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

        # Create the model
        model = self.get_model(dataset)

        # Fit the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Evaluate the model performance.
        if dataset == 'flowers':
            accuracy = accuracy_score(y_test, predictions)
            print('Classifier accuracy {:.2f}'.format(accuracy))
        else:
            rmse = sqrt(mean_squared_error(y_test, predictions))
            print('Regression root mean squared error {:.2f}'.format(rmse))

    @staticmethod
    def get_model(dataset):
        """Train a classifier or a regression model with a sklearn algorithm.
        Note that there are MANY hyperparameters you pass into these models.
        Refer to the online sklearn docs for more information."""
        if dataset == 'flowers':
            return GradientBoostingClassifier(random_state=RANDOM_SEED)
        else:
            return GradientBoostingRegressor(random_state=RANDOM_SEED)


if __name__ == "__main__":
    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    flower_data = datasets.load_iris(return_X_y=True)
    housing_data = datasets.load_boston(return_X_y=True)

    # Predict with the two models and the two datasets.
    predictor = StochasticGradientBoostingDemo()
    predictor.make_prediction(flower_data, 'flowers')
    predictor.make_prediction(housing_data, 'housing')
