from algothon2021prediction import Prediction
import numpy as np
from sklearn import (
    preprocessing,
    svm,
    feature_selection,
    neural_network,
    pipeline,
    decomposition,
    model_selection,
)


class SimpleNNPredictor(Prediction):
    def __init__(self, **params):
        super(SimpleNNPredictor, self).__init__(**params)

        self.scaler = preprocessing.StandardScaler()
        self.normalizer = preprocessing.Normalizer()
        self.pca = decomposition.PCA(n_components=15, random_state=123)

        self.model = neural_network.MLPRegressor(
            hidden_layer_sizes=(100, 100, 100, 100), activation="relu", random_state=123
        )

        self.pipeline = pipeline.make_pipeline(self.model)

    def train(self, X_train, y_train):
        # Fit scaler and transform
        X_train = self.scaler.fit_transform(X_train)
        # Fit PCA and transform
        X_train = self.pca.fit_transform(X_train)
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)

    def predict(self, X) -> float:
        # Transform data by previously fitted scaler
        X = self.scaler.transform(X)
        # Transform data by previously fitted PCA
        X = self.pca.transform(X)
        return self.pipeline.predict(X)
