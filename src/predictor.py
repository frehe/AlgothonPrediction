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
from scipy import stats


class SimpleNNPredictor(Prediction):
    def __init__(self, **params):
        super(SimpleNNPredictor, self).__init__(**params)

        self.scaler = preprocessing.StandardScaler()
        self.normalizer = preprocessing.Normalizer()
        self.pca = decomposition.PCA(n_components=20, random_state=123)

        self.model = neural_network.MLPRegressor(
            hidden_layer_sizes=(200, 400, 400, 400, 400, 200, 100),
            activation="relu",
            random_state=123,
            solver="adam",
            alpha=0.005,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=True,
            validation_fraction=0.2,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10,
            max_fun=15000,
        )
        # self.model = svm.SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1)

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
