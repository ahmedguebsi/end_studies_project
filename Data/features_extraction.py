"""
Class which extracts features for a given dataframe's epoch

PE - special entropy - calculated by applying the Shannon function to the normalized power spectrum based on the peaks of a Fourier transform
AE - Approximate entropy - calculated in time domain without phase-space reconstruction of signal (short-length time series data)
SE - Sample entropy - similar to AE. Se is less sensitive to changes in data length with larger values corresponding to greater complexity or irregularity in the data
FE - Fuzzy entropy - stable results for different parameters. Best noise resistance using fuzzy membership function.

"""
from math import ceil, floor
from typing import Dict, List, Optional, Tuple, TypedDict

import antropy as an
import EntropyHub as eh
import numpy as np
from pandas import DataFrame, Series
from scipy import signal
from scipy.signal.spectral import periodogram

from environment import FREQ


def fuzzy_entropy(x):
    return eh.FuzzEn(x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]


def sample_entropy(x):
    return an.sample_entropy(x)


# don't normalize because you have to normalze across all drivers and not based on 1 driver and 1 sample
def spectral_entropy(x, freq: float):
    axis = -1
    sf = freq
    normalize = False

    x = np.asarray(x)
    _, psd = periodogram(x, sf, axis=axis)
    psd_norm = psd[1:] / psd[1:].sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


def approximate_entropy(x):
    return an.app_entropy(x, order=2)


def psd_welch(x: Series, fs=FREQ):
    _, psd = signal.welch(x, fs=fs)
    return psd


class FeatureContext(TypedDict):
    freq_signal: Optional[float]
    freq_filter_range: Optional[Tuple[float, float]]
    epoch_id: Optional[int]


class FeatureExtractorFeatureContextError(Exception):
    pass


class FeatureExtractorFeatureInvalidArgument(Exception):
    pass


class FeatureExtractor:
    def __init__(self, selected_feature_names: List[str]):
        self._set_mappers()
        self.signal = None
        self.freq = None
        filtered_features = filter(lambda pair: pair[0] in selected_feature_names, self._name_to_function_mapper.items())
        self.selected_features_functions = list(map(lambda pair: pair[1], filtered_features))

    def _set_mappers(self):
        self._name_to_function_mapper = {
            "mean": self.feature_mean,
            "std": self.feature_standard_deviation,
            "psd": self.feature_power_spectral_density,
            "PE": self.feature_spectral_entropy,
            "AE": self.feature_approximate_entropy,
            "SE": self.feature_sample_entropy,
            "FE": self.feature_fuzzy_entropy,
        }
        self._feature_function_to_mapper_mapper = {v: k for k, v in self._name_to_function_mapper.items()}

    def _validate_feature_context(self, key, context: FeatureContext):
        if key not in context:
            raise FeatureExtractorFeatureContextError("Missing key '{}' in the context.".format(key))

    def fit(self, signal, freq=None):
        self.signal = signal
        self.freq = freq

    def feature_mean(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: np.mean(x), axis=0)

    def feature_standard_deviation(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: np.std(x), axis=0)

    def feature_power_spectral_density(self, df: DataFrame, freq_filter_range: Optional[Tuple[float, float]], epoch_id: int, **kwargs):
        pdfs = df.apply(func=lambda x: psd_welch(x, self.freq), axis=0)
        low_freq, high_freq = freq_filter_range
        pdfs = pdfs[floor(low_freq) : ceil(high_freq)]
        return pdfs.mean(axis=0)

    def feature_spectral_entropy(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: spectral_entropy(x.to_numpy(), self.freq), axis=0)

    def feature_approximate_entropy(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: approximate_entropy(x.to_numpy()), axis=0)

    def feature_sample_entropy(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: sample_entropy(x.to_numpy()), axis=0)

    def feature_fuzzy_entropy(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: fuzzy_entropy(x.to_numpy()), axis=0)

    def get_features(self, df: DataFrame, **kwargs: FeatureContext) -> Dict:
        features = {}
        for feature_function in self.selected_features_functions:
            feature_name = self.function_to_name(feature_function)
            features[feature_name] = feature_function(df, **kwargs)
        return features

    def name_to_function(self, feature_name):
        return self._name_to_function_mapper[feature_name]

    def function_to_name(self, feature_function):
        return self._feature_function_to_mapper_mapper[feature_function]

    def get_feature_names(self):
        return list(map(lambda feature_function: self.function_to_name(feature_function), self.selected_features_functions))


if __name__ == "__main__":
    pass