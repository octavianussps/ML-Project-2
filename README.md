# Machine Learning Project 2 Extract roads from satellite images

## **Authors**
  * Marion Chabrier - marion.chabrier@epfl.ch
  * Valentin Margraf - valentin.margraf@epfl.ch
  * Octavianus Sinaga - octavianus.sinaga@epfl.ch

This readme file contains useful information of the structure of the project containing the code and report. For further information of the project and its results we advise you to read the report.

The detailed explanation of the project is on the report (`latexreport/report.pdf`).

  

## **Project structure**



| Folder  | Files |
|:--:|:--:|
| `test_set_images/`  | test data |
| `out/`  | contains the final subission file, also csv.format |
| `latexreport/` | contains the pdf and the latex report of our project |
| `scripts/`  | all the python scripts we used in this project, further explanation below |

In `scripts/` we can see:

+ `run.py` is a script which produces the submission.csv file for the test data.
+ `mask_to_submission.py` is a script to make a submission file from a binary image
+ `submission_to_mask.py` is a script to reconstruct an image from the sample submission file

  

## **Description**

This project was part of a challenge from EPFL course : Machine Learning CS-433 and was hosted on the platform AIcrowd.
They are a set of satellite/aerial images acquired from GoogleMaps and ground-truth images where each pixel is labeled as {road, background}. The objective is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each patch of pixel size 16x16.

## **Prerequisites**
The code is tested with the following versions 
- `Python 3.7.x`
- `Numpy 1.17.xx`
- `Matplotlib 3.1.x`
- `TensorFlow 2.0`
- `Pillow 6.2.xx`


## **Dataset**
In https://www.aicrowd.com/challenges/epfl-ml-road-segmentation you will find the training set consisting of images with their ground truth and the test set 
Download it and put it in the folder named `Data`.


## **Running**
In order to submit the predictions we give on the test data, you have to run the `run.py` file. It will load the data, preprocess it, build the feature matrix and then train the model (in this case Least Squares). Then the model is used to predict the labels of the test data and the `out/submission.csv` file will be generated.

  

It will give some output like this:


```
loading data

preprocessing data

building polynomial with degree 11

training model with least squares

predicting labels for test data

exporting csv file
```

## **Results**

We achieved a categorical accuracy of 0.822 and a F1- score of  0.728 on the website. All the details are in the report.
