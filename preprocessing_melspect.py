import numpy as np
import librosa
import math
import json
import pandas as pd
import os
import ast
import warnings
from tqdm.notebook import tqdm
import argparse

##### GLOBAL VARIABLES #####

SAMPLE_RATE=22050 # Always this way. 
N_FFT = 2048 # Recommended for music signals (vs speech) https://librosa.org/doc/main/generated/librosa.stft.html
HOP_LENGTH=512 # In general N_FFT // 4 https://librosa.org/doc/main/generated/librosa.stft.html#librosa.stft
N_CATEGORIES = 16
SAVE_CUTS_SEPARATELY = False # set to "False" if you want to save place and not save all cuts mfcc separately
SIGNAL_TIME_30 = SAMPLE_RATE*30 # default lenght of a signal of 30sec
SIGNAL_TIME_27 = SAMPLE_RATE*27 # min. size required to work with the sample
CUTS_SECONDS = [3, 10] # the different cut you want to produce (here 3 and 10 secs cuts)
metadata_path='Datasets/fma_metadata/'
fma_dir = 'Datasets/fma_medium/'
save_dir = 'Datasets/preprocess_mfcc/'


##### METHODS #####

def load_FMA(filepath: str):
    """ loads FMA Dataset (metadata)

    Args:
        filepath (str): the path to the metadata folder + the file you want to read

    Returns:
        _type_: fma metadata information based on the file you want to read (features, echonest, genres, tracks)
    """
    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                "category", categories=SUBSETS, ordered=True
            )
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


def get_genre_mapping(genres):
    """ assigns the correct genre to each track
    """
    #reindex on 'title' for quick search
    genres.reset_index(drop=False).set_index('title')
    #select only top genres (parent==0) and sort them alphabetically
    top_genres = genres[genres['parent'] == 0].sort_index(ascending=True)
    id_to_title = top_genres.reset_index(drop=False).to_dict()['title']
    title_to_id = {v: k for k, v in id_to_title.items()}
    return title_to_id


def prepare_folders(save_dir: str, cuts=False, cuts_seconds=CUTS_SECONDS):
    """ prepares the Dataset folder by creating the different subfolders need to store the preprocessed datasets.
    For each dataset we have training, validation and test folder and each of this folder contains 16 folder representing the 16 genres

    Args:
        save_dir (str): path to save the data
        cuts (_type_, optional): True if you want to cut the sample into different size. Defaults to False.
        cuts_seconds (_type_, optional): The list of the size (in sec) you want to build a dataset for. Defaults to CUTS_SECONDS.
    """
    if 'full_30sec' not in os.listdir(save_dir):
        os.mkdir(save_dir+'full_30sec')
    if cuts:
        for cut in cuts_seconds:
            if 'cut'+str(cut)+'s' not in os.listdir(save_dir):
                os.mkdir(save_dir+'cut'+str(cut)+'s')
    for time_cut in ['full_30sec'] + (['cut'+str(cut)+'s' for cut in cuts_seconds] if cuts else []):
        for _set in ['training', 'validation', 'test']:
            if _set not in os.listdir(save_dir+time_cut):
                os.mkdir(save_dir+time_cut+'/' + _set)
            for _class in range(N_CATEGORIES):
                if str(_class) not in os.listdir(save_dir+time_cut+'/'+ _set):
                    os.mkdir(save_dir+time_cut+'/'+ _set + '/' + str(_class))


def preprocess_melspect(fma_dir: str,
                        save_dir: str,
                        tracks: pd.DataFrame,
                        genre_mapping: dict,
                        cuts=False,
                        sample_rate=SAMPLE_RATE,
                        hop_length=HOP_LENGTH):
    """ extensive method to preprocess the FMA dataset and generate the melspctograms for each tracks and snippets according to the needed cuts

    Args:
        fma_dir (str): path to fma directory
        save_dir (str): path to the save directory
        tracks (pd.DataFrame): Pandas DataFrame with all the tracks
        genre_mapping (dict): dictionnary giving the genre for each tracks
        cuts (bool, optional): True if you want to divide the 30secs tracks into multiple snippets of different length. Defaults to False.
        sample_rate (_type_, optional): sample rate use for melspectogram generation. Defaults to SAMPLE_RATE.
        hop_length (_type_, optional): hop length use for melspectrogram generation. Defaults to HOP_LENGTH.

    Raises:
        Exception: if error during the preprocessing process
    """

    assert fma_dir.endswith('/')

    def save_mfcc_file(file_name, _set, data, label, cut='None', iteration=False, iteration_suffix=""):
        """ adds one dimension to the data and save it in .npy format

        Args:
            file_name (_type_): _description_
            _set (_type_): training, validation 
            data (_type_): the melspectrogram data
            label (_type_): the genre/label of the data
            cut (str, optional): _description_. Defaults to 'None'.
            iteration (bool, optional): True if you want to save each iteration of a cut. Defaults to False.
            iteration_suffix (str, optional): suffix used to save a specific iteration. Defaults to "".
        """
        data = np.expand_dims(data, axis=2)
        if cut == 'None':
            f_save = save_dir + 'full_30sec/'+str(_set) + '/' + str(label) + '/' + file_name + '.npy'
            np.save(f_save, data)
        else:
            # WARNING: add save per_cut option
            if iteration:
                f_save = save_dir + 'cut' + str(cut) + 's/'+str(_set) + '/' + \
                        str(label) + '/' + file_name + "_" + iteration_suffix + '.npy'
                np.save(f_save, data)
            else:
                f_save = save_dir + 'cut' + str(cut) + 's/' + str(_set) + '/'  + \
                         str(label) + '/' + file_name + '.npy'
                np.save(f_save, data)

    def cutting_handler(signal, f_name, _set, genre_top_id, cuts_seconds, sample_rate=SAMPLE_RATE,hop_length=HOP_LENGTH ):
        """ handles the case where we save melspectrograms representing only a snippet/cut of the original 30sec track

        Args:
            signal (_type_): the signal from librosa
            f_name (_type_): track name
            _set (_type_): the set we are working on (validation, training, testing)
            genre_top_id (_type_): the genre of the track
            cuts_seconds (_type_): the list of the different cuts you want to do
            sample_rate (_type_, optional): Defaults to SAMPLE_RATE.
            hop_length (_type_, optional): Defaults to HOP_LENGTH.
        """

        for cut in cuts_seconds:
            # how many cuts can we do, lower bound (because of int()) (math.ceil)
            iterations = int((len(signal) / sample_rate) / cut)
            data_cut = {
                'label': genre_top_id,
                'mfcc': []
            }
            for i in range(iterations):
                signal_cut = signal[i * cut * SAMPLE_RATE: (i + 1) * cut * SAMPLE_RATE]
                spect = librosa.feature.melspectrogram(y=signal_cut, sr=sample_rate, n_fft=N_FFT,
                                                       hop_length=hop_length)
                mfcc = librosa.power_to_db(spect, ref=np.max).T

                save_mfcc_file(f_name, _set, mfcc, genre_top_id, cut=cut, iteration=True, iteration_suffix=str(i))

    # first prepare the folders
    prepare_folders(save_dir, cuts=cuts,  cuts_seconds=CUTS_SECONDS)

    # list to save files that give error
    error_files = []

    # read track, process them and save them
    for folder in tqdm(os.listdir(fma_dir)):
        if folder.isdigit():
            for file in os.listdir(fma_dir + folder):
                f_path = fma_dir+folder+'/'+file
                f_name = file.replace('.mp3', '')
                genre_top = tracks.loc[int(f_name)]['track']['genre_top']
                # assert track file, long enough (>27sec) and with top_genre (48%~ of the songs roughly)
                if f_name.isdigit() and os.path.getsize(f_path) > 5e4 \
                    and type(genre_top) != float:
                    genre_top_id = genre_mapping[genre_top]
                    _set = tracks['set', 'split'].loc[int(f_name)]
                    try:
                        signal, sr = librosa.load(fma_dir + folder + '/' + file, sr=sample_rate)
                        # cut the signal to 30s if longer...
                        if len(signal) > SIGNAL_TIME_30:
                            signal = signal[0:SIGNAL_TIME_30]
                        #...drop it if less than 27s...
                        elif len(signal) < SIGNAL_TIME_27:
                            raise Exception("Sorry, sample id " + f_name + " is less than 27sec")
                        #... otherwise pad it to 30s by looping over the beginning of the track
                        else:
                            signal = np.append(signal, signal[0:SIGNAL_TIME_30 - len(signal)])
                        assert len(signal) == SIGNAL_TIME_30

                        # compute the melspectrogram and scale it to dB, transposing to get the time component as first component
                        spect = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=N_FFT,
                                                               hop_length=hop_length)
                        mfcc = librosa.power_to_db(spect, ref=np.max).T

                        #save it as npy
                        save_mfcc_file(f_name, _set, mfcc, genre_top_id)

                        #handle cuts
                        if cuts:
                            cutting_handler(signal, f_name, _set, genre_top_id, cuts_seconds=CUTS_SECONDS)

                    except:
                        print('Error with file ' + f_name)


def main(args):
    """ runs the whole preprocessin pipeline
    """
    if args.verbose:
        print("Load FMA metadata...")
    tracks = load_FMA(metadata_path+'tracks.csv')
    genres = load_FMA(metadata_path+'genres.csv')
    if args.verbose:
        print("Process mining...")
    genre_mapping = get_genre_mapping(genres)
    if args.verbose:
        print("Preprocessing ...")
    preprocess_melspect(fma_dir, save_dir, tracks, genre_mapping, cuts=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='takes raw data, outputs preprocessed data')
    parser.add_argument('-v', '--verbose', dest='verbose', help='do you want to print information', action='store_true')
    args = parser.parse_args()
    main(args)

