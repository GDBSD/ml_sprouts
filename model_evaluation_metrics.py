# -*- coding: utf-8 -*-

"""Examples of various model validation approaches.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_24_Chapter.xhtml
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Make the models deterministic
RANDOM_SEED = 42


class ValidateModel:
    """In this module, we survey more common metrics for evaluating regression and classification
    models. Other metrics are explored in the other sample modules.Note: we don't include validation
    interpretations. Read the docs.
    """

    def validate_model(self, data):
        """Demonstrate common metrics for regression and classification use cases and how to
        implement them using Scikit-learn.

        :param data:  tuple - a tuple containing the data and the targets.
        :return:
        """

        X = data[0]
        y = data[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=RANDOM_SEED)
        print()
        self.validate_linear_regression_model(X_train, X_test, y_train, y_test)
        print('=' * 50)
        self.evaluate_linear_regression_with_cross_validation(X, y)
        print('=' * 50)
        self.evaluate_logistic_regression_model(X, y)
        print('=' * 50)

    @staticmethod
    def validate_linear_regression_model(X_train, X_test, y_train, y_test):
        """Evaluate a linear regression model with a variety of approaches.

        :param X_train: Numpy array - features (training data)
        :param X_test:  Numpy array - targets (training data)
        :param y_train:  Numpy array - features (test data)
        :param y_test: Numpy array - targets (test data)
        :return:
        """

        print('Validating linear regression model ...')

        # Normalize the dataset before fitting the model.
        linear_reg = LinearRegression(normalize=True)

        # Fit the model on the training set.
        linear_reg.fit(X_train, y_train)

        # Make predictions on the test set.
        predictions = linear_reg.predict(X_test)

        # Evaluate the model performance using mean square error metric.
        print('Linear regression mean squared error: {:.2f}'.format(
            mean_squared_error(y_test, predictions)))

        # Evaluate the model performance using mean absolute error metric.
        print('Linear regression mean absolute error: {:.2f}'.format(
            mean_absolute_error(y_test, predictions)))

        # evaluate the model performance using r-squared error metric
        print("R-squared score: %.2f" % r2_score(y_test, predictions))
        print('Linear regression r-squared score: {:.2f}'.format(
            r2_score(y_test, predictions)))

    @staticmethod
    def evaluate_linear_regression_with_cross_validation(X, y):
        """An example of regression evaluation metrics implemented with cross-validation.
        The MSE and MAE metrics for cross-validation are implemented with the sign inverted.
        The simple way to interpret this is to have it in mind that the closer the values are
        to zero, the better the model.

        :param X: Numpy array - features
        :param y: Numpy array - labels
        :return:
        """
        print('Evaluating linear regression with cross validation ...')

        # Initialize KFold to shuffle the data before splitting.
        kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        # Create the model.
        linear_reg = LinearRegression(normalize=True)

        # Fit the model using cross validation - score with Mean square error (MSE)
        mse_cv_result = cross_val_score(linear_reg, X, y, cv=kfold, scoring='neg_mean_squared_error')

        print('Linear regression negative mean squared error: {:.2f}'.format(
            mse_cv_result.mean(), mse_cv_result.std()))

        # Fit the model using cross validation - score with Mean absolute error (MAE)
        mae_cv_result = cross_val_score(linear_reg, X, y, cv=kfold, scoring='neg_mean_absolute_error')

        print('Linear regression negative mean absolute error: {:.2f}'.format(
            mae_cv_result.mean(), mae_cv_result.std()))

        # Fit the model using cross validation - score with R-squared
        r2_cv_result = cross_val_score(linear_reg, X, y, cv=kfold, scoring='r2')

        print('Linear regression r-squared score: {:.2f}'.format(
            r2_cv_result.mean(), r2_cv_result.std()))

    @staticmethod
    def evaluate_logistic_regression_model(X, y):
        """Validate a logistic regression classification  model implemented with
        cross validation.

        :param X: Numpy array - features
        :param y: Numpy array - labels
        :return:
        """

        print('Validating logistic regression model ...')

        # Initialize KFold and shuffle the data.
        kfold = KFold(n_splits=3, shuffle=True)

        # Create the model
        logistic_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)

        # Fit the model using cross validation - score with accuracy
        accuracy_cv_result = cross_val_score(logistic_reg, X, y, cv=kfold, scoring="accuracy")

        print('Logistic regression accuracy: {:.2f}'.format(
            accuracy_cv_result.mean(), accuracy_cv_result.std()))

        # Fit the model using cross validation - score with Log-Loss
        logloss_cv_result = cross_val_score(logistic_reg, X, y, cv=kfold, scoring="neg_log_loss")

        print('Logistic regression log-Loss likelihood: {:.2f}'.format(
            logloss_cv_result.mean(), logloss_cv_result.std()))


if __name__ == "__main__":
    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    flower_data = datasets.load_iris(return_X_y=True)

    validator = ValidateModel()
    validator.validate_model(flower_data)
