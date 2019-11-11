# -*- coding: utf-8 -*-

"""Example of the feature_union pipeline.
O'Reilly E-book page:
https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/463852_1_En_24_Chapter.xhtml
"""

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union

# Make the models deterministic
RANDOM_SEED = 42


class FeatureUnionPipeline:
    """Scikit-learn provides a module for merging the output of several transformers called feature_union.
    It does this by fitting each transformer independently to the dataset, and then their respective outputs
    are combined to form a transformed dataset for training the model.

    FeatureUnion works in the same way as a Pipeline, and in many ways can be thought of as a means of building
    complex pipelines within a Pipeline."""

    @staticmethod
    def create_pipeline(data):
        """Combine the output of recursive feature elimination (RFE) and PCA for feature engineering, then
        apply the Stochastic Gradient Boosting (SGB) ensemble model for regression to train the model.

        :param data:  tuple - a tuple containing the data and the targets.
        :return:
        """

        X = data[0]
        y = data[1]

        # # Construct a pipeline for feature engineering - make_union() is similar to make_pipeline()
        rfe = RFE(estimator=RandomForestRegressor(n_estimators=100), n_features_to_select=6)
        pca = PCA(n_components=9)
        feature_union = make_union(rfe, pca)

        # Build the pipeline model
        pipe = make_pipeline(
            feature_union,
            GradientBoostingRegressor(n_estimators=100))

        # Run the pipeline
        kfold = KFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
        cv_result = cross_val_score(pipe, X, y, cv=kfold)

        print('Accuracy: {:.2f}'.format(
            cv_result.mean() * 100.0, cv_result.std() * 100.0))


if __name__ == "__main__":
    # Get some sample data from sklearn datasets. Setting return_X_y to True will
    # constrain the output to be a tuple containing only the data and the targets.
    housing_data = datasets.load_boston(return_X_y=True)

    pipeline_builder = FeatureUnionPipeline()
    pipeline_builder.create_pipeline(housing_data)
