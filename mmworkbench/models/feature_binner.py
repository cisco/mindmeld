import numpy as np

ZERO = 1e-20


class FeatureMapper(object):
    def __init__(self):
        self.feat_name = None
        self.values = []
        self.std = None
        self.mean = None

        self.std_bins = []

        self._num_std = 2
        self._size_std = 0.5

    def add_value(self, value):
        self.values.append(value)

    def fit(self):
        self.std = np.std(self.values)
        self.mean = np.mean(self.values)

        range_start = self.mean - self.std * self._num_std
        num_bin = 2 * int(self._num_std / self._size_std)
        bins = [range_start]

        while num_bin > 0 and self.std > ZERO:
            range_start += self.std * self._size_std
            bins.append(range_start)
            num_bin -= 1
        self.std_bins = np.array(bins)

    def map_bucket(self, value):
        return np.searchsorted(self.std_bins, value)


class FeatureBinner(object):
    def __init__(self):
        self.features = {}

    def fit(self, X_train):
        """
        Get necessary information for each bucket

        Args:
            X_train (list of list of list): training data
        """
        for sentence in X_train:
            for word in sentence:
                for feat_name, feat_value in word.items():
                    self._collect_feature(feat_name, feat_value)

        for feat, mapper in self.features.items():
            mapper.fit()

    def transform(self, X_train):
        """
        Get necessary information for each bucket

        Args:
            X_train (list of list of dict): training data
        """
        new_X_train = []
        for sentence in X_train:
            new_sentence = []
            for word in sentence:
                new_word = {}
                for feat_name, feat_value in word.items():
                    new_feats = self._map_feature(feat_name, feat_value)
                    if new_feats:
                        new_word.update(new_feats)
                new_sentence.append(new_word)
            new_X_train.append(new_sentence)
        return new_X_train

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

    def _collect_feature(self, feat_name, feat_value):
        try:
            feat_value = float(feat_value)
        except Exception:
            return
        mapper = self.features.get(feat_name, FeatureMapper())
        mapper.feat_name = feat_name
        mapper.add_value(feat_value)

        self.features[feat_name] = mapper

    def _map_feature(self, feat_name, feat_value):
        try:
            feat_value = float(feat_value)
        except Exception:
            return {feat_name: feat_value}
        if feat_name not in self.features:
            return {feat_name: feat_value}

        mapper = self.features[feat_name]
        new_feat_value = mapper.map_bucket(feat_value)
        return {feat_name: new_feat_value}
