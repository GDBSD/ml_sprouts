# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression

"""Examples of various feature engineering approaches.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_24_Chapter.xhtml
"""

# Make the models deterministic
RANDOM_SEED = 42


class FeatureEngineering:

    def select_features(self, algorithms_data, feature_reduction):
        """Review some techniques implemented in Scikit-learn for selecting relevant features
        from a dataset.

        :param algorithms_data: list - list of tuples, e containing the algorithm name and the data.
        :param feature_reduction: float - ratio of features to keep to total features.
        :return:
        """

        methods_dict = {'k_best_chi': self.calculate_k_best_chi_scores, 'rfe': self.apply_rfe,
                        'feat_imp': self.apply_feature_importance}

        for a in algorithms_data:
            algorithm = a[0]
            training_data = a[1]
            features = len(training_data[0][1])
            features_to_keep = round(features * feature_reduction)
            method_to_call = methods_dict[algorithm]
            method_to_call(training_data, features_to_keep)
            print('=' * 25)
            print()

    @staticmethod
    def calculate_k_best_chi_scores(data, no_features):
        """Use chi-squared stats of non-negative features, chi2 (classification).

        :param data: tuple - a tuple containing the features and the labels.
        :param no_features: int - numbers of features to keep
        :return:
        """
        X = data[0]
        y = data[1]

        print('Running calculate_k_best_chi_scores to select best {} features ...'.format(no_features))

        # Produce a new Numpy array with the four best features
        X_new = SelectKBest(chi2, k=no_features).fit_transform(X, y)

        print('Best chi-squared results')
        print('New data has {} features.'.format(X_new.shape[1]))

    @staticmethod
    def apply_rfe(data, no_features):
        """Recursive feature elimination (RFE) recursively removes features, builds a
        model using the remaining attributes and calculates model accuracy. RFE is able to
        work out the combination of attributes that contribute to the prediction on the
        target variable (or class)

        :param data: tuple - a tuple containing the features and the labels.
        :param no_features: int - numbers of features to keep
        :return:
        """
        X = data[0]
        y = data[1]

        print('Running apply_rfe to select best {} features ...'.format(no_features))

        linear_reg = LinearRegression()
        rfe = RFE(estimator=linear_reg, n_features_to_select=no_features)
        rfe.fit(X, y)
        print('RFE results:')

        # ref.support_ returns an array with boolean values to indicate whether an attribute
        # was selected using RFE.
        print(rfe.support_)

        # ref.ranking_ returns an array with positive integer values to indicate the attribute
        # ranking with a lower score indicating a higher ranking.
        print(rfe.ranking_)

    @staticmethod
    def apply_feature_importance(data, no_features):
        """Drop irrelevant features in the dataset using the sklearn SelectFromModel module.

        :param data: tuple - a tuple containing the features and the labels.
        :param no_features: int - Not used by this algorithm.
        :return:
        """
        X = data[0]
        y = data[1]

        print('Running apply_feature_importance ...')

        classifier = AdaBoostClassifier()
        classifier.fit(X, y)
        classes = classifier.classes_
        importances = classifier.feature_importances_

        print('classes:', classes)
        print('AdaBoost feature importances: {}'.format(importances))


if __name__ == "__main__":
    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    flower_data = datasets.load_iris(return_X_y=True)
    housing_data = datasets.load_boston(return_X_y=True)

    # Specify the ratio of features you want to keep from the source data.
    features_target = 0.75

    # algorithms_to_eval = [('k_best_chi', flower_data), ('rfe', housing_data), ('feat_imp', flower_data)]
    algorithms_to_eval = [('k_best_chi', flower_data), ('rfe', housing_data), ('feat_imp', flower_data)]

    selector = FeatureEngineering()
    selector.select_features(algorithms_to_eval, features_target)
