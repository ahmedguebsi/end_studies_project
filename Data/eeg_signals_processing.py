from math import floor
from typing import Callable, List, Optional, Tuple, TypedDict, Union

import numpy as np
from mne import Epochs
from mne.io import read_raw_cnt
from mne.io.base import BaseRaw
from pandas import DataFrame


from preprocess import df_replace_values
from environment import (FATIGUE_STR, FREQ, LOW_PASS_FILTER_RANGE_HZ,
                       NOTCH_FILTER_HZ, NUM_USERS,
                       SIGNAL_DURATION_SECONDS_DEFAULT, SIGNAL_OFFSET,
                       channels_good, driving_states, feature_names,
                       get_brainwave_bands)
from feature_extraction import FeatureExtractor
from utils_file_saver import save_df
from helper_functions import (get_cnt_filename, glimpse_df, is_arg_default,
                             serialize_functions)
from paths import PATH_DATAFRAME, PATH_DATASET_CNTg
from utils_signal import SignalPreprocessor


def load_clean_cnt(filename: str, channels: List[str]):
    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.pick_channels(channels)
    return cnt


def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    """ Reduces signal's lenght by cropping it """
    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.crop(tmin=start, tmax=end)


def signal_filter_notch(signal: BaseRaw, filter_hz):
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))


def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    """ Filters the signal with lower and higher pass-band edge by using zero-phase filtering """
    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)


def epochs_to_dataframe(epochs: Epochs, drop_columns=["time", "condition"]):
    """ Converts to dataframe and drops unnecessary columns """
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(drop_columns, axis=1)
    return df

def get_column_names(channels, feature_names, preprocess_procedure_names : List[str]):
    prod= product(channels, feature_names, preprocess_procedure_names)
    return list(map(lambda strs:"_".join(strs), prod))

feature_extractor = FeatureExtractor(selected_feature_names=feature_names)
signal_preprocessor = SignalPreprocessor()

"""
4 signal preprocessing procedures with SignalPreprocessor
"standard" -> .notch and .filter with lower and higher pass-band edge defined in the research paper
"AL", "AH", "BL", "BH" -> .notch and .filter with lower and higher pass-band edge defined in brainwave_bands in env.py
"reref" -> .notch and .filter with lower and higher pass-band edge defined in the research paper and rereference within electodes
"""


filter_frequencies = {"standard": LOW_PASS_FILTER_RANGE_HZ}
if use_brainbands:
    filter_frequencies.update(get_brainwave_bands())


""" Registers preprocessing procedures that will filter frequencies defined in filter_frequencies"""
for freq_name, freq_range in filter_frequencies.items():
    low_freq, high_freq = freq_range

    procedure = serialize_functions(
        lambda s, notch_filter_hz=NOTCH_FILTER_HZ: signal_filter_notch(s, notch_filter_hz),
        lambda s, low_freq=low_freq, high_freq=high_freq: s.copy().filter(low_freq, high_freq),
        lambda s, freq=FREQ, signal_offset=SIGNAL_OFFSET, signal_duration=signal_duration: signal_crop(s, freq, signal_offset, signal_duration),
    )

    signal_preprocessor.register_preprocess_procedure(freq_name, procedure=procedure, context={"freq_filter_range": freq_range})

""" Registers preprocessing procedure that uses channel rereferencing"""
if use_reref:
    low_freq, high_freq = LOW_PASS_FILTER_RANGE_HZ
    proc = serialize_functions(
        lambda s, notch_filter_hz=NOTCH_FILTER_HZ: signal_filter_notch(s, notch_filter_hz),
        lambda s, low_freq=low_freq, high_freq=high_freq: s.filter(low_freq, high_freq).set_eeg_reference(ref_channels="average", ch_type="eeg"),
        lambda s, freq=FREQ, signal_offset=SIGNAL_OFFSET, signal_duration=signal_duration: signal_crop(s, freq, signal_offset, signal_duration),
    )
    signal_preprocessor.register_preprocess_procedure("reref", proc, context={"freq_filter_range": LOW_PASS_FILTER_RANGE_HZ})

training_cols = get_column_names(channels, feature_extractor.get_feature_names(), signal_preprocessor.get_preprocess_procedure_names())
df_dict = {k: [] for k in ["is_fatigued", "driver_id", "epoch_id", *training_cols]}

for driver_id, driving_state in tqdm(list(product(range(0, driver_num), driving_states))):

    is_fatigued = 1 if driving_state == FATIGUE_STR else 0
    signal_filepath = str(Path(PATH_DATASET_CNT, get_cnt_filename(driver_id + 1, driving_state)))

    signal = load_clean_cnt(signal_filepath, channels)
    signal_preprocessor.fit(signal)
    for proc_index, (signal_processed, proc_name, proc_context) in tqdm(enumerate(signal_preprocessor.get_preprocessed_signals())):
        epochs = make_fixed_length_epochs(signal_processed, verbose=False)
        df = epochs_to_dataframe(epochs)

        freq_filter_range = proc_context["freq_filter_range"]
        feature_extractor.fit(signal_processed, FREQ)
        for epoch_id in tqdm(range(0, signal_duration)):
            """
            Filter the dataframe rows by selecting the rows with epoch_id.
            get_features caculates all features for a given epoch
                * each feature is caculated for all channels
                * for 30 channels the shape of a feature will be (30,) (if feature is e.g. float)
                * feature_dict dictionary contains all features
            Features are flattened to a combination (channel, feature, preprocess procedure) which defines a single column
                * e.g. F4_std_standard
                * total number of columns is (len(features) * len(channels) * len(preprocess_procedures))
            For loop appends the result to each column. Effectively, it's itterating through the feature array (30,) and appends each float to appropriate column
                * once the for loop is completed, an appropriate result will be appended to each column
            """

            df_epoch = df.loc[df["epoch"] == epoch_id, channels].head(epoch_events_num)
            feature_dict = feature_extractor.get_features(df_epoch, epoch_id=epoch_id, freq_filter_range=freq_filter_range)

            for channel_idx, channel in enumerate(channels):
                for feature_name, feature_array in feature_dict.items():
                    df_dict[get_column_name(feature_name, channel, proc_name)].append(feature_array[channel_idx])
            if proc_index == 0:
                df_dict["epoch_id"].append(epoch_id)
                df_dict["driver_id"].append(driver_id)
                df_dict["is_fatigued"].append(is_fatigued)


if __name__ == "__main__":
    pass
