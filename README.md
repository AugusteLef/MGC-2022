# Music Genre Classification Project
The purpose of this repository is to offer an overview of the methods used during our project and the possibility to reproduce our experiments and results presented in the report.

## File Structure
Some scripts may assume the following file-structure (you might have to create missing directories):

### Directories
- ***Datasets/*** : Directory containing all training-, test- and preprocessed data (and original data)
- ***Datasets/fma_medium/*** : Directory containing all the original data from the FMA dataset
- ***Datasets/fma_metadata/*** : Directory containing all the metadata of the FMA dataset
- ***Datasets/preprocess_mfcc/*** : Directory containg 3 subfolders with 30s, 10s and 3s cuts after pre-processing and folder preparation
- ***Models/*** : Directory containing all the model you train (we add some pre-trained models from our experiments FYI)
- ***Figures/*** : Directory containing the confusion matrix and the training history (loss and accuracy) in .png format evaluation
- ***Results/*** : Directory containing .txt files with the accuracy of each previously evaluated models

### Scripts
- ***preprocessing_melspect.py*** : Script running the preprocessing pipeline
- ***training.py*** : Script allowing to train any models presented in the paper
- ***evaluate.py*** : Script allowing to evaluate any models and save .png and .txt of the results in the corresponding directory

### Additionals
*Datasets/, Models/, Figures/, and Results/* are empty directories at the beginning.

# Melspectrogram and CRNN methods
## Datasets

[All the (preprocessed)-datasets used for our experiments were too large to be added on [Polybox](https://polybox.ethz.ch/). Therefore in order to run the experiment you will first have to download the original datasets and to run the preprocessing script]

Download the [FMA](https://github.com/mdeff/fma) dataset, and the metadata:

1. [fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip): 25,000 tracks of 30s, 16 unbalanced genres (22GiB)
2. [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)

Move them to the *Datasets/* directory:
```
unzip fma_medium.zip

mv fma_medium/* Datasets

unzip fma_metadata.zip

mv fma_metadata/* Datasets
```
The *Datasets/fma_metadata/* directory should contain the following files:
- ***tracks.csv***: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
- ***genres.csv***: all 163 genres with name and parent (used to infer the genre hierarchy and top-level genres).
- ***features.csv***: common features extracted with [librosa](https://librosa.org/doc/latest/index.html).
- ***echonest.csv***: audio features provided by Echonest (now Spotify) for a subset of 13,129 tracks.

the *Datasets/fma_medium/* directory should contain the following files
- ***156 folders***: each cointaining tracks in .mp3 format

## Running Our Experiments

Guidelines for running our experiments are presented here. We assume that the git-directory has been cloned, that the correct file structure has been set up (i.e. adding missing directories according to description above) and that the datasets have been downloaded and put in the *Datasets/* directory.

### Further Preparations

Create and start virtual environment:
```bash
python3 -m venv venv
source activate venv
```
Install dependencies (make sure to be in venv):
```bash
pip install -r requirements.txt
```

### Preprocessing

Before running the preprocessing, ensure that *Datasets/* contains de following directory:
- ***fma_medium/***
- ***fma_metadata/***
- ***preprocess_mfcc/***

You may need to create the last folder yourself with the following command line:
```bash
cd Dadasets/
mkdir preprocess_mfcc
cd ..
```

You are now ready to run the preprocessing script that will build 30sec, 10sec and 3sec datasets with the correct architecture needed for the rest of the project. To do so, run the following command line.
```
python3 preprocessing_melspect.py
```

Note that we set up *preprocessing_melspect.py* to reproduce the exact experiments we performed during the project. However, if you want to try different cuts, modify the hyperparameters used for the melspectrogram generation and more; you can modify the global varibale at the top of the preprocessing file.
Disclaimer: we ensured that the code is reliable for 10s and 3s cuts, using other cut lengths might lead to some kind of error in the process.

### Training
Using *training.py* you can train any models we used during our experiments. To make the process easier we build the script such that it takes different arguments allowing you to train different model architectures :

- ***"-30sec" or "-10sec" or "-3sec"*** : chose if you want to train the model with 30, 10 or 3 seconds samples (mandatory)
- ***"-4c" or "-3c" or "-3c"*** : chose the number of convolution block in the model (mandatory)
- ***"-l1" or "-l2"*** : chose the regulirisation loss you want to use, l1 loss or l2 loss (optional)
- ***"-lrs"*** : if you want to use the learning rate scheduler (optional)
- ***"-gru2"*** : if you want add a second consecutive GRU layer (optional)
- ***"-ep20"*** : if you want to run only 20 epoch instead of the 50 epoch (optional)

When the training is done, the model as well as the history of the training process are stored in the *Models/* directory.


Example to train a model for 30sec samples with 4 convolution blocks, l2 loss, learning rate scheduler, 2 consecutive gru, 20 epochs: 
```
python3 training.py -30sec -4c -l2 -lrs -gru2 -ep20
```

If you want to run the model from the [CRNN for Music Classification paper by Keunwoo Choi & al.](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&sortby=pubdate&citation_for_view=ZrqdSu4AAAAJ:ULOm3_A8WrAC) you need to use the specific argument:

- ***"-papermodel"***

This argument can be combine ONLY with the size of the sample you want to use.

Example: train the model from the paper with 30sec samples:
```
python3 training.py -30sec -papermodel
```

Again, note that we ensure that the script allow you to reproduce our exact experiments. If you want to try other cut lengths or achitecture you might need to modfify the script according to your needs. Also, you might want to use different batchsize or epochs (we forced 32 and 50/20 as we obtained best results with this configuration). To do so you can modify the global variables at the beginning of the script.

### Evaluation
Once a model has been trained, you can now evaluate it. We offer a script that:

1. Evaluate a given model and save the results in a .txt
2. Allow you to use our voting system (Divide and Conquer) on models train with 10s and 3s samples
3. Save the confusion matrix and the training history (loss and accuracy) in .png format

The script takes as inputs the following arguments:

- ***"model_name"*** : the name of the model you want to evaluate aka. the name of the model's directory saved after the training step (mandatory)
- ***"-30sec", "-10sec" or "-3sec"*** : the sample's size used to trained the model you want to evaluate (mandatory)
- ***"-voting"*** : if you want to apply the voting methods (divide and conquer) --> only possible for models trained with 10s or 3s samples (optional)


Example: evaluate the model trained with 4 convolution blocks on 20 epochs with 10sec samples, and using the voting (divide and conquer) method:
```
python3 evaluate.py "4conv_20epochs" -10 -voting
```

Warning: as we plot the different figures in time, you might have to close them to continue the process

# Final Note

If you want to know more about our preprocessing methods, model architectured, results and more; please refer to our report. You can aslo have a look at our code (available in this repository) for a better understanding of the different processes that has been executed.

@authors
Auguste, Marc and Lukas
