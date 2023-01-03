import json
import numpy as np
import os
import sys
import pandas as pd
import ast
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import argparse
import pickle
from tqdm.notebook import tqdm
from tensorflow import keras

##### GLOBAL VARIABLES #####
pathmodel_30sec = "Models/30sec/"
pathmodel_10sec = "Models/10sec/"
pathmodel_3sec = "Models/3sec/"
pathdata_30sec = "Datasets/preprocess_mfcc/full_30sec/test"
pathdata_10sec = "Datasets/preprocess_mfcc/cut10s/test"
pathdata_3sec = "Datasets/preprocess_mfcc/cut3s/test"
path_figure = "Figures/"
path_results = "Results/"
metadata_path='Datasets/fma_metadata/'


##### METHODS #####
def history_val_loss(history, model_name: str):
    """ Plot and save the training/validation loss and accuracy of the model

    Args:
        history (_type_): the history of the trained model
        model_name (str): model name use to save the figure
    """
    plt.subplot(211)
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="lower right")

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper right")
    plt.tight_layout()
    plt.savefig(path_figure + model_name + "_historyplot.png")
    plt.show()

def load_FMA(filepath: str):
    """ loads FMA Dataset (metadata)

    Args:
        filepath (str): the path to the metadata folder

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

def normalize_array(arr):
    """ quick normalization (min-max)

    Args:
        arr (_type_): the array to be normalized

    Returns:
        _type_: the normalized array
    """
    min_val = -80
    diff_val = 80
    return (arr - (min_val))/diff_val

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

def voting_predict(model, tracks):
    """ Perform evaluation of the model and apply the voting method (divide and conquer)

    Args:
        model (_type_): model we want to evaluate
        tracks (_type_): list of tracks used for the evaluation

    Returns:
        _type_: prediction
    """
    predict = np.expand_dims(np.zeros(16), axis=0).T
    aggreg_predictions = np.expand_dims(np.zeros(16), axis=0)
    weight = 1/len(tracks)
    for t in tracks:
        prediction = model.predict(np.array(t),verbose = 0)
        aggreg_predictions += prediction * weight
    predict[aggreg_predictions.argmax()]=1
    return predict.flatten()

def evaluate_and_confusion_matrix_ensemble(x_test, y_test, model, genres, model_name):
    """ Plot and save confusion matrix adn return the accuracy of the model after applying the voting method

    Args:
        x_test (_type_): test set samples
        y_test (_type_): test set true value
        model (_type_): keras model
        genres (_type_): genres of tracks
        model_name (_type): the name of the model

    Returns:
        _type_ : accuracy of the model with voting method
    """
    accuracy = 0
    genre_map = {v: k for k, v in get_genre_mapping(genres).items()}

    y_true = tf.argmax(y_test, axis=1)
    predict=[]
    for i in tqdm(range(len(x_test))):
        pred_i = voting_predict(model, x_test[i])
        predict.append(pred_i)
        accuracy += 1 if pred_i.argmax() == y_true[i].numpy() else 0
    y_pred = tf.argmax(predict, axis=1)
    
    # Have to be 1-D vectors. Not one-hot encoded
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=genre_map.values(), yticklabels=genre_map.values(),
                annot=True, fmt='g', ax=ax, cmap="viridis", vmax=200)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Label')
    plt.savefig(path_figure + model_name +"_voting_confusionmatrix.png")
    plt.show()
    return accuracy/len(y_true)

def confusion_matrix_normal(x_test, y_test, model, genres, model_name):
    """ Plot and save confusion matrix adn return the accuracy of the model

    Args:
        x_test (_type_): test set sample
        y_test (_type_): test set true value
        model (_type_): keras model
        genres (_type_): genre of tracks
        model_name (_type_): the name of the model
    """
    genre_map = {v: k for k, v in get_genre_mapping(genres).items()}

    y_true = tf.argmax(y_test, axis=1)
    y_pred = tf.argmax(model.predict(x_test), axis=1)

    # Have to be 1-D vectors. Not one-hot encoded
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=genre_map.values(), yticklabels=genre_map.values(),
                annot=True, fmt='g', ax=ax, cmap="viridis", vmax=200)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Label')
    plt.savefig(path_figure + model_name +"_confusionmatrix.png")
    plt.show()

def load_data_ensemble(_path, split):
    """ Load the data in the correct way such that we can then perform voting method and do the evaluation smoothly

    Args:
        _path (_type_): path to the data
        split (_type_): the sample length (10 or 3)

    Returns:
        _type_: _description_
    """
    assert split==3 or split==10
    x_test, y_test = [],[]
    for genre_i in tqdm(os.listdir(_path)):
        if str(genre_i).isdigit():
            files = os.listdir(_path+ "/" +genre_i)
            track_ids = list(set([f.split('_')[0] for f in files]))
            for file in track_ids:
                if file.isdigit():
                    splits = 10 if split==3 else 3 
                    arr=[]
                    for cut in range(splits):
                        f = _path+"/" +genre_i+"/"+file+ '_'+str(cut)+'.npy'              
                        arr_cut = normalize_array(np.load(f))
                        arr.append(np.swapaxes(arr_cut.T,1,2))
                    x_test.append(arr)
                    y_test.append([1 if i ==int(genre_i) else 0 for i in range(16)])
    return np.array(x_test), np.array(y_test)

def load_data_normal(_path):
    x_test, y_test = [],[]
    for genre_i in tqdm(os.listdir(_path)):
        if str(genre_i).isdigit():
            for file in os.listdir(_path+'/'+genre_i):
               track_id = file.split('.')[0].split('_')[0]
               if track_id.isdigit():
                  f = _path+'/'+genre_i+"/"+file               
                  arr = np.load(f)
                  x_test.append(arr)
                  y_test.append([1 if i ==int(genre_i) else 0 for i in range(16)])

    x_test, y_test = normalize_array(np.array(x_test)), np.array(y_test)
    return x_test, y_test


def main(args):

    print("hello")
    if args.verbose:
        print("Running the evaluation..")
        print("params:")
        print(args)

    if args.s30sec and args.voting:
        print("error: you want to use voting (divide and conquer) method on 30sec samples. This is not doable. Chose 10sec or 3sec samples instead.")

    if (args.s30sec and args.s10sec) or (args.s30sec and args.s3sec) or (args.s10sec and args.s3sec):
        print("error: multiple size (sec) selected. Please chose only 1")
        return

    genres = load_FMA(metadata_path+'genres.csv')

    path_to_model = ""
    path_to_history = ""
    if args.s30sec:
        path_to_model = pathmodel_30sec + args.model_name
        path_to_history = pathmodel_30sec + "history_" + args.model_name
    if args.s10sec:
        path_to_model = pathmodel_10sec + args.model_name
        path_to_history = pathmodel_10sec + "history_" + args.model_name
    if args.s3sec:
        path_to_model = pathmodel_3sec + args.model_name
        path_to_history = pathmodel_3sec + "history_" + args.model_name

    isExist = os.path.exists(path_to_model)
    if not isExist:
        print("error: the model name does not exist")

    model = keras.models.load_model(path_to_model)
    history = []

    with (open(path_to_history, "rb")) as openfile:
        history = pickle.load(openfile)

    accuracy = None
    if args.voting:
        if args.s10sec:
            x_test, y_test = load_data_ensemble(pathdata_10sec, split=10)
            history_val_loss(history, "10sec_" + args.model_name)
            accuracy = evaluate_and_confusion_matrix_ensemble(x_test, y_test, model, genres, "10sec_" + args.model_name)
            with open(path_results + "voting_10sec_" + args.model_name + ".txt", 'w') as f:
                f.write(str(accuracy))        
        elif args.s3sec:
            x_test, y_test = load_data_ensemble(pathdata_3sec, split=3)
            history_val_loss(history, "3sec_" + args.model_name)
            accuracy = evaluate_and_confusion_matrix_ensemble(x_test, y_test, model, genres, "3sec_" + args.model_name)
            with open(path_results + "voting_3sec_" + args.model_name + ".txt", 'w') as f:
                f.write(str(accuracy))    
    else:
        if args.s30sec:
            x_test, y_test = load_data_normal(pathdata_30sec)
            history_val_loss(history, "30sec_" + args.model_name)
            confusion_matrix_normal(x_test, y_test, model, genres, "30sec_" + args.model_name)
            accuracy = model.evaluate(x=x_test, y=y_test)
            with open(path_results + "30sec_" + args.model_name + ".txt", 'w') as f:
                f.write(str(accuracy))    
        elif args.s10sec:
            x_test, y_test = load_data_normal(pathdata_10sec)
            history_val_loss(history, "10sec_" + args.model_name)
            confusion_matrix_normal(x_test, y_test, model, genres, "10sec_" + args.model_name)
            accuracy = model.evaluate(x=x_test, y=y_test)
            with open(path_results + "10sec_" + args.model_name+  ".txt", 'w') as f:
                f.write(str(accuracy))   
        elif args.s3sec:
            x_test, y_test = load_data_normal(pathdata_3sec)
            history_val_loss(history, "3sec_" + args.model_name)
            confusion_matrix_normal(x_test, y_test, model, genres, "3sec_" + args.model_name)
            accuracy = model.evaluate(x=x_test, y=y_test)
            with open(path_results + "3sec_" + args.model_name + ".txt", 'w') as f:
                f.write(str(accuracy))

    if not (args.s30sec or args.s10sec or args.s3sec):
        print("error: no sample/snippet size selected. Please chose 1")
        return

    if args.verbose:
        print("... Evaluation done") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='takes a model as input and performs the evaluation step')

    parser.add_argument('model_name', type=str, help='model name', action='store')

    parser.add_argument('-30sec', '--30sec', dest='s30sec', help='do you want to evaluate a model with 30sec sample?', action='store_true')
    parser.add_argument('-10sec', '--10sec', dest='s10sec', help='do you want to evaluate a model with 10sec sample?', action='store_true')
    parser.add_argument('-3sec', '--3sec', dest='s3sec', help='do you want to evaluate a model with 3sec sample?', action='store_true')
    
    parser.add_argument('-voting', '--voting', dest='voting', help="do you want to use the voting (divide and conquer) method?", action='store_true')

    parser.add_argument('-v', '--verbose', dest='verbose', help="do you want to print information",action='store_true')
    
    args = parser.parse_args()
    main(args)

