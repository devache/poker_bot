
import abc

class Model(abc):

    self.model = None

    def _feature_logging(self, features):
        pass
    def _outcome_logging(self, outcome):
        pass
    def transform(self, features):

        self._feature_logging(features)
        outcome = self.model.transform(features)
        self._outcome_logging(outcome)

        return outcome

    def fit(self, result):
        pass