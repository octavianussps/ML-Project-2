# Machine Learning Project 2 Extract roads from satellite images

## **Authors**
  * Marion Chabrier - marion.chabrier@epfl.ch
  * Valentin Margraf - valentin.margraf@epfl.ch
  * Octavianus Sinaga - octavianus.sinaga@epfl.ch

This readme file contains useful information of the structure of the project containing the code and report. For further information of the project and its results we advise you to read the report.

The detailed explanation of the project is on the report (`latex/report.pdf`).


## **Project structure**

| Folder  | Files |
|:--:|:--:|
| `data/`  | test data |
| `latex/` | contains the pdf and the latex report of our project |
| `models/`  | the models we use |
| `out/`  | contains the final submission files, also csv format |
| `src/`  | all the python scripts we used in this project, further explanation below |

In `src/` you can see:

+ `run.py` is a script which produces the submission.csv file for the test data.
+ `cnn_model.py` contains several functions to build the cnn 
+ `helpers.py` contains several helper functions in order to run the cnn 
+ `train.py` is a script to train the cnn using the training data 
+ `visualisation` is a folder containing some notebooks to help for the visualization

  

## **Description**

This project was part of a challenge from EPFL course Machine Learning CS-433 and was hosted on the platform AIcrowd.
There are a set of satellite/aerial images acquired from GoogleMaps and ground-truth images where each pixel is labeled as {road, background}. The objective is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each patch of pixel size 16x16.

## **Prerequisites**

The code is tested with the following versions 
- `Python 3.7.x`
- `Numpy 1.17.xx`
- `Matplotlib 3.1.x`
- `TensorFlow 2.0`
- `Pillow 6.2.xx`


## **Dataset**
We did not put the trainig data in the github because it takes too much memory place.
In https://www.aicrowd.com/challenges/epfl-ml-road-segmentation you will find the training set consisting of images with their ground truth and the test set where we did our predictions on. 
Download it and put it in the folder named `data`.


## **Running**
In order to submit the predictions we give on the test data, you have to run the `run.py` file. It will build the model and load the trained weights. Then the model will predict on the test data and the `out/submission.csv` file will be generated.


## **Results**

We achieved a F1- score of 0.882 and the accuracy of 0.938 on the website. All the details are in the report.
