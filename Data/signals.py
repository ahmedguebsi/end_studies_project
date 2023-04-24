from math import floor
from typing import Callable, List, Optional, Tuple, TypedDict, Union

import numpy as np
from mne import Epochs
from mne.io import read_raw_cnt
from mne.io.base import BaseRaw
from pandas import DataFrame


def load_clean_cnt(filename: str, channels: List[str]):
    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.pick_channels(channels)
    return cnt


def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.crop(tmin=start, tmax=end)


def signal_filter_notch(signal: BaseRaw, filter_hz):
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))


def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)


def epochs_to_dataframe(epochs: Epochs, drop_columns=["time", "condition"]):
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(drop_columns, axis=1)
    return df


class SignalPreprocessorContext(TypedDict):
    freq_filter_range: Optional[Tuple[float, float]]


class SignalPreprocessorProcedureDict(TypedDict):
    procedure: Callable[[DataFrame], DataFrame]
    context: SignalPreprocessorContext
    name: Union[str, None]


class SignalPreprocessorException(Exception):
    pass


class SignalPreprocessorSignalNotSet(SignalPreprocessorException):
    pass


class SignalPreprocessorInvalidName(SignalPreprocessorException):
    pass


class SignalPreprocessor:
    DEFAULT_PREPROCESSING_PROCEDURE = {"name": None, "procedure": lambda x: x, "context": {}}

    def __init__(self, signal=None):
        self.preprocess_procedures: List[SignalPreprocessorProcedureDict] = []
        self.signal: Union[BaseRaw, None] = signal

    def fit(self, signal: BaseRaw):
        self.signal = signal

    def _check_signal_fit(self):
        if self.signal is None:
            raise SignalPreprocessorSignalNotSet("Signal is not set.")

    def _get_procedure_dict_tripplet(self, procedure_dict: SignalPreprocessorProcedureDict):
        preprocessor_procedure = procedure_dict["procedure"]
        procedure_name = procedure_dict["name"]
        procedure_context = procedure_dict["context"]
        return preprocessor_procedure, procedure_name, procedure_context

    def get_preprocessed_signals(self):
        """ Generator which yields all preprocessed procedures applied to the fitted signal."""
        self._check_signal_fit()
        if self.preprocess_procedures == []:
            preprocessor_procedure, procedure_name, procedure_context = self._get_procedure_dict_tripplet(self.DEFAULT_PREPROCESSING_PROCEDURE)
            signal_preprocessed = preprocessor_procedure(self.signal)
            yield signal_preprocessed, procedure_name, procedure_context

        for procedure_dict in self.preprocess_procedures:
            preprocessor_procedure, procedure_name, procedure_context = self._get_procedure_dict_tripplet(procedure_dict)
            signal_preprocessed = preprocessor_procedure(self.signal.copy())
            yield signal_preprocessed, procedure_name, procedure_context

    def register_preprocess_procedure(self, procedure_name: str, procedure, context: SignalPreprocessorContext = {}):
        if not isinstance(procedure_name, str):
            raise SignalPreprocessorInvalidName("Procedure name has to be a string.")

        procedure_dict = dict(name=procedure_name, procedure=procedure, context=context)
        self.preprocess_procedures.append(procedure_dict)
        return procedure_dict

    def unregister_preprocess_procedure(self, procedure_name):
        return self.preprocess_procedures.pop(procedure_name, None)

    def clear_preprocess_procedures(self):
        self.preprocess_procedures = []
        return

    def get_preprocess_procedure_names(self) -> List[str]:
        return list(map(lambda x: x["name"], self.preprocess_procedures))


if __name__ == "__main__":
    print("success")