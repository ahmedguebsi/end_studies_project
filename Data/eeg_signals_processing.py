from itertools import product
from math import floor
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
#from mne.io import read_raw_cnt
import numpy as np
from mne import Epochs
from mne.epochs import make_fixed_length_epochs
from mne.io.base import BaseRaw
from mne.io.cnt import read_raw_cnt
from pandas import DataFrame, set_option
from tqdm import tqdm

from environment import (FATIGUE_STR, FREQ, USE_REREF, LOW_PASS_FILTER_RANGE_HZ,
                              NOTCH_FILTER_HZ, SIGNAL_OFFSET,sig_channels,
                              channels_good, driving_states, feature_names,
                              get_brainwave_bands, NUM_USERS, SIGNAL_DURATION_SECONDS_DEFAULT)
from features_extraction import FeatureExtractor
from helper_functions import (get_cnt_filename, glimpse_df, serialize_functions,isnull_any, rows_with_null)
from signals import SignalPreprocessor
#from preprocess import (df_replace_values)
import mne
from mne.preprocessing import ICA

driver_num = NUM_USERS
signal_duration = SIGNAL_DURATION_SECONDS_DEFAULT
epoch_events_num = FREQ
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
use_brainbands = False
use_reref = USE_REREF
channels_ignore=[]
#channels = channels_good
#channels = list(set(channels_good) - set(channels_ignore))

channels = sig_channels
channels = list(set(sig_channels) - set(channels_ignore))
is_complete_dataset=True
train_metadata = {"is_complete_dataset": is_complete_dataset, "brains": use_brainbands, "reref": use_reref}

PATH_DATASET_CNT= r"C:\Users\Ahmed Guebsi\Downloads\cnt"



def df_replace_values(df: DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def apply_ica(cnt_file):
    # Read in EEG data
    raw = mne.io.read_raw_cnt(cnt_file)
    raw.interpolate_bads()

    # Apply band-pass filter to EEG data
    raw.filter(1, 30)

    # Initialize ICA object and fit ICA to EEG data
    ica = ICA(n_components=20, random_state=42)
    ica.fit(raw)

    # Plot ICA components
    ica.plot_components()

    hi_cut = 40
    ica.plot_properties(epochs, picks=range(0, ica.n_components_), psd_args={'fmax': hi_cut});
    # Apply ICA to EEG data
    ica.apply(raw)

    return raw


def load_clean_cnt(filename: str, channels: List[str]):
    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.interpolate_bads()
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

def get_column_names(channels, feature_names, preprocess_procedure_names : List[str]):
    prod= product(channels, feature_names, preprocess_procedure_names)
    return list(map(lambda strs:"_".join(strs), prod))

def get_column_name(feature: str, channel: str, suffix: Union[str, None] = None):
    result = "_".join([channel, feature])
    result = result if suffix is None else "_".join([result, suffix])
    return result

feature_extractor = FeatureExtractor(selected_feature_names=feature_names)
signal_preprocessor = SignalPreprocessor()




filter_frequencies = {"standard": LOW_PASS_FILTER_RANGE_HZ}
if use_brainbands:
    filter_frequencies.update(get_brainwave_bands())


""" preprocessing procedures that will filter frequencies defined in filter_frequencies"""
for freq_name, freq_range in filter_frequencies.items():
    low_freq, high_freq = freq_range

    procedure = serialize_functions(
        lambda s, notch_filter_hz=NOTCH_FILTER_HZ: signal_filter_notch(s, notch_filter_hz),
        lambda s, low_freq=low_freq, high_freq=high_freq: s.copy().filter(low_freq, high_freq),
        lambda s, freq=FREQ, signal_offset=SIGNAL_OFFSET, signal_duration=signal_duration: signal_crop(s, freq, signal_offset, signal_duration),
    )

    signal_preprocessor.register_preprocess_procedure(freq_name, procedure=procedure, context={"freq_filter_range": freq_range})

training_cols = get_column_names(channels, feature_extractor.get_feature_names(), signal_preprocessor.get_preprocess_procedure_names())
df_dict = {k: [] for k in ["is_fatigued", "driver_id", "epoch_id", *training_cols]}

for driver_id, driving_state in tqdm(list(product(range(0, driver_num), driving_states))):

    is_fatigued = 1 if driving_state == FATIGUE_STR else 0
    signal_filepath = str(Path(PATH_DATASET_CNT, get_cnt_filename(driver_id + 1, driving_state)))

    signal = load_clean_cnt(signal_filepath, channels)
    signal_preprocessor.fit(signal)
    for proc_index, (signal_processed, proc_name, proc_context) in tqdm(enumerate(signal_preprocessor.get_preprocessed_signals())):
        # By default epoch duration = 1
        epochs = make_fixed_length_epochs(signal_processed, verbose=False)

        #epochs = mne.Epochs(signal_processed, epochs, tmin=0, tmax=1, baseline=None, detrend=1, preload=True)
        print(type(epochs))
        # Remove eye blink artifacts using ICA
        ica = mne.preprocessing.ICA(n_components=5, random_state=42)
        ica.fit(epochs)

        # identified using visual inspection
        #ica.exclude = [0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
        epochs.load_data()
        epochs_clean = ica.apply(epochs)
        #ica.plot_components()

        df = epochs_to_dataframe(epochs_clean)
        #df = epochs_to_dataframe(epochs)

        freq_filter_range = proc_context["freq_filter_range"]
        feature_extractor.fit(signal_processed, FREQ)
        for epoch_id in tqdm(range(0, signal_duration)):
            df_epoch = df.loc[df["epoch"] == epoch_id, channels].head(epoch_events_num)
            feature_dict = feature_extractor.get_features(df_epoch, epoch_id=epoch_id, freq_filter_range=freq_filter_range)

            for channel_idx, channel in enumerate(channels):
                for feature_name, feature_array in feature_dict.items():
                    df_dict[get_column_name(feature_name, channel, proc_name)].append(feature_array[channel_idx])
            if proc_index == 0:
                df_dict["epoch_id"].append(epoch_id)
                df_dict["driver_id"].append(driver_id)
                df_dict["is_fatigued"].append(is_fatigued)



"""Create dataframe from rows and columns"""
df = DataFrame.from_dict(df_dict)
df["is_fatigued"] = df["is_fatigued"].astype(int)
df["driver_id"] = df["driver_id"].astype(int)
df["epoch_id"] = df["epoch_id"].astype(int)
glimpse_df(df)
print(isnull_any(df))
print(rows_with_null(df))
df = df_replace_values(df)
print(isnull_any(df))
print(rows_with_null(df))
df.to_pickle(str(Path(output_dir, ".clean_raw_df.pkl")))



if __name__ == "__main__":
    pass
