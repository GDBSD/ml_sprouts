# -*- coding: utf-8 -*-

"""Examples of some common model tuning approaches.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_24_Chapter.xhtml
"""

from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Make the models deterministic
RANDOM_SEED = 42


class ModelTuner:

    def tune_model(self, data):
        """Demonstrate Scikit-learn GridSearchCV and RandomizedSearchCV for tuning model
        hyperparameters.

        :param data:  tuple - a tuple containing the data and the targets.
        :return:
        """

        X = data[0]
        y = data[1]

        print()
        self.apply_gridsearch(X, y)
        print('=' * 50)
        self.apply_random_search(X, y)
        print('=' * 50)

    @staticmethod
    def apply_gridsearch(X, y):
        """Grid search comprehensively explores all the specified hyper-parameter values for an estimator.

        :param X: Numpy array - features
        :param y: Numpy array - labels
        :return:
        """

        print('Applying GridSearch to find the best hyperparameters ...')

        # Construct grid search parameters in a dictionary
        parameters = {'n_estimators': [2, 4, 6, 8, 10, 12, 14, 16],
                      'max_depth': [2, 4, 6, 8],
                      'min_samples_leaf': [1, 2, 3, 4, 5]}

        # Create the model
        rf_model = RandomForestRegressor()

        # Run the grid search.
        #   n_estimators is the number of trees in the forest.
        #   max_depth is the maximum depth of the tree.
        #   min_samples_leaf is the minimum number of samples required to split an internal node.
        #   cv sets the cross-validation splitting strategy.
        #   iid - if True, return the average score across folds, weighted by the number of samples in each test set.
        grid_search = GridSearchCV(estimator=rf_model, param_grid=parameters, cv=5, iid=False)

        # Fit the model
        grid_search.fit(X, y)

        print('Best GridSearch accuracy: {:.3f}'.format(
            grid_search.best_score_ * 100.0))

    @staticmethod
    def apply_random_search(X, y):
        """As opposed to grid search, not all the provided hyper-parameter values are evaluated, but
        rather a determined number of hyper-parameter values are sampled from a random uniform distribution.

        :param X: Numpy array - features
        :param y: Numpy array - labels
        :return:
        """

        print('Applying randomized search to find the best hyperparameters ...')

        # Construct grid search parameters in a dictionary
        parameters = {'n_estimators': [2, 4, 6, 8, 10, 12, 14, 16],
                      'max_depth': [2, 4, 6, 8],
                      'min_samples_leaf': [1, 2, 3, 4, 5]}

        # Create the model.
        rf_model = RandomForestRegressor()

        # Run the grid search.
        randomized_search = RandomizedSearchCV(
            estimator=rf_model, param_distributions=parameters, n_iter=10, cv=5, iid=False, random_state=RANDOM_SEED)

        # Fit the model.
        randomized_search.fit(X, y)

        # Best set of hyper-parameter values:
        print('Best n_estimators: {}'.format(randomized_search.best_estimator_.n_estimators))
        print('Best max_depth: {}'.format(randomized_search.best_estimator_.max_depth))
        print('Best min_samples_leaf: {}'.format(randomized_search.best_estimator_.min_samples_leaf))

        # Accuracy:
        print('Best random search accuracy: {:.3f}'.format(
            randomized_search.best_score_ * 100.0))


if __name__ == "__main__":
    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    housing_data = datasets.load_boston(return_X_y=True)

    tuner = ModelTuner()
    tuner.tune_model(housing_data)
