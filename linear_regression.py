# -*- coding: utf-8 -*-

from sklearn.linear_model import Ridge
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt


class LinearRegressionDemo:
    """A simple linear regression model.
    O'Reilly E-book page:
    https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_19_Chapter.xhtml
    """

    def make_prediction(self, data, add_polynomial=False, alpha=0):
        """Train a linear regression model and evaluate is quality.

        :param data: tuple - a tuple containing the data and the targets.
        :param add_polynomial: boolean - create higher-order polynomials from the dataset.
        :param alpha float - controls the magnitude of the regularization parameter.
        :return:
        """

        split_data = self.split_data(data, add_polynomial)
        X_train = split_data['X_train']
        X_test = split_data['X_test']
        y_train = split_data['y_train']
        y_test = split_data['y_test']

        # Instantiate a linear regression model with regularization.
        linear_reg = Ridge(alpha=alpha)

        # Fit the model on the training set.
        linear_reg.fit(X_train, y_train)

        # Make predictions on the test set.
        predictions = linear_reg.predict(X_test)

        # Fit the model on the training set.
        linear_reg.fit(X_train, y_train)
        # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)

        # Evaluate the model performance using the root mean square error metric.
        if add_polynomial:
            print('Root mean squared error (RMSE) with added polynomial term: {:.2f}'.format(
                sqrt(mean_squared_error(y_test, predictions))))
        else:
            print('Root mean squared error (RMSE) without added polynomial term: {:.2f}'.format(
                sqrt(mean_squared_error(y_test, predictions))))

    @staticmethod
    def split_data(data, add_polynomial):
        """Split the data into train and test datasets. If add_polynomial = True
        create higher-order polynomials from the dataset features in hope of fitting
        a more flexible model that may better capture the variance in the dataset.
        It is rare to find datasets from real-world events where the features have a
        perfectly underlying linear structure. So adding higher-order terms is most
        likely to improve the model performance. But we must watch out to avoid overfitting.

        :param data: tuple - a tuple containing the data and the targets.
        :param add_polynomial: boolean - create higher-order polynomials from the dataset.
        :return: dictionary with the split datasets
        """

        # Separate features and target.
        X = data[0]
        y = data[1]
        if add_polynomial:
            # Create higher-order polynomials from the dataset features in hope of fitting
            # a more flexible model that may better capture the variance in the dataset.
            polynomial_features = PolynomialFeatures(2)
            X_higher_order = polynomial_features.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_higher_order, y, shuffle=True)
            split_data = {'X_train': X_train, 'X_test': X_test,
                          'y_train': y_train, 'y_test': y_test}
            return split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
            split_data = {'X_train': X_train, 'X_test': X_test,
                          'y_train': y_train, 'y_test': y_test}
            return split_data


if __name__ == "__main__":

    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    sample_data = datasets.load_boston(return_X_y=True)

    # The higher the value of alpha (i.e., the regularization parameter), the more restricted
    # the coefficients (or weights) of the cost function. Hence, if the value of λ is high,
    # the model can result in a learning bias (i.e., it underfits the dataset).
    # However, if the value of alpha approaches zero, the regularization parameter has negligible
    # effects on the model, hence resulting in overfitting the model.
    ridge_alpha = 1

    model = LinearRegressionDemo()

    model.make_prediction(sample_data, add_polynomial=False, alpha=ridge_alpha)
    model.make_prediction(sample_data, add_polynomial=True, alpha=ridge_alpha)


