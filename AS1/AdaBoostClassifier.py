import joblib
from BaseLearner import BaseLearner
from sklearn.ensemble import AdaBoostClassifier

class AdaBoostModel(BaseLearner):


    def __init__(self, X_train, X_test, y_train, y_test, pipe, pre_processed_feature_names, class_names, dataset_name):

        self.model = AdaBoostClassifier(random_state=1)

        super().__init__(X_train, X_test, y_train, y_test, pipe, self.model, pre_processed_feature_names, class_names, dataset_name)

        self.model.fit(self.X_train, self.y_train)
        self.model_params = {'random_state': 1}

    def fit(self):
        super().model.fit(self.X_train, self.X_test)

    def predict(self, y):
        super().model.predict(y)

    def update_and_refit_model(self):

        self.model = AdaBoostClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)

    def find_hyper_params(self):
        raise NotImplemented

    def save_model(self, filename):
        joblib.dump(self, filename + "pkl")