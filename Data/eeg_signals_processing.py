from math import floor
from typing import Callable, List, Optional, Tuple, TypedDict, Union

import numpy as np
from mne import Epochs
from mne.io import read_raw_cnt
from mne.io.base import BaseRaw
from pandas import DataFrame


def load_clean_cnt(filename: str, channels: List[str]):
    """
    Load Raw CNT (EEG signal) and pick given channels
    """

    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.pick_channels(channels)
    return cnt


def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    """
    Reduces signal's lenght by cropping it
    """

    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.crop(tmin=start, tmax=end)


def signal_filter_notch(signal: BaseRaw, filter_hz):
    """
    Apply notch filter to the signal
    """
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))


def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    """
    Filters the signal with lower and higher pass-band edge by using zero-phase filtering
    """

    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)


def epochs_to_dataframe(epochs: Epochs, drop_columns=["time", "condition"]):
    """
    Converts to dataframe and drops unnecessary columns
    """
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(drop_columns, axis=1)
    return df