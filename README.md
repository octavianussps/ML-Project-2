# Machine Learning Project 2 Extract roads from satellite images

## **Authors**
  * Marion Chabrier - marion.chabrier@epfl.ch
  * Valentin Margraf - valentin.margraf@epfl.ch
  * Octavianus Sinaga - octavianus.sinaga@epfl.ch

This readme file contains useful information of the structure of the project containing the code and report. For further information of the project and its results we advise you to read the report.

The detailed explanation of the project is on the report (`latex/report.pdf`).

## **Description**

This project was part of a challenge from EPFL course Machine Learning CS-433 and was hosted on the platform AIcrowd.
There are a set of satellite/aerial images acquired from GoogleMaps and ground-truth images where each pixel is labeled as {road, background}. The objective is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each patch of pixel size 16 x 16.

We implement a Convolutional Neural Network using the sliding window Approach and a U-Net in order totackle this task. The U-Net gave us a F1-score of 0.898 as the best result.

## **Project structure**

| Folder  | Files |
|:--:|:--:|
| `data/`  | test data |
| `latex/` | contains the pdf and the latex report of our project |
| `models/`  | the differents models we use |
| `out/`  | contains the final submission files, also csv format |
| `src/`  | all the python scripts we used in this project, further explanation below |

### **In `src/` you can see:**

+ `run.py` is a script which produces the `final_UNET_submission.csv` file for the test data with the UNET model
+ `unetmodel_training.py` is a script to train the UNET using the training data 
+ `unetmodel.py` contains several functions to build the unet model 
+ `predictCNN.py` is a script which produces the `final_CNN_submission.csv` file for the test data with the CNN model
+ `cnnmodel_training.py` is a script to train the cnn using the training data 
+ `cnn_model.py` contains several functions to build the cnn model
+ `helpers.py` contains several helper functions in order to run the cnn 

+ `visualisation` is a folder containing some notebooks to help for the visualization

### **In `models/` you can see:**
+ `unetLReLU.h5` : the final U-NET model with Leaky ReLU
+ `weightsFinal.h5` : the final CNN model
+ `weightsSimpleNetworkBigPatch.h5` : the CNN model with less filters in each layers
  
### **In `out/` you can see:**
+ `final_UNET_submission.csv` : the final U-NET submission files with Leaky ReLU with F1-score of 0.898 and the accuracy of 0.947
+ `final_CNN_submission.csv` : the final CNN submission files with F1-score of 0.882 and the accuracy of 0.938
+ `final_small_CNN_submission.csv` : the CNN submission files with less filters in each layers with F1-score of 0.862 and the accuracy of 0.924


## **Prerequisites**

The code is tested with the following versions 
- `Python 3.7.x`
- `Numpy 1.17.xx`
- `Matplotlib 3.1.x`
- `TensorFlow 2.0`
- `Pillow 6.2.xx`

### **Installation**
Now, install the necessary data science libraries. Make sure to install them in order listed below.

```
conda install ipython
conda install jupyter
conda install -c conda-forge matplotlib
conda install pandas
conda install scipy
conda install scikit-learn
conda install -c anaconda pillow
pip install --upgrade tensorflow
pip install --upgrade keras
```


## **Dataset**
We did not put the trainig data in the github because it takes too much memory place.
In https://www.aicrowd.com/challenges/epfl-ml-road-segmentation you will find the training set consisting of images with their ground truth and the test set where we did our predictions on. 
Download it and put it in the folder named `data/`.


## **Running**
In order to submit the predictions we give on the test data, you have to run the `run.py` file. It will run the unet model. It will load the unet model with the trained weights already build in `cnnmodel_training.py` . Then the model will predict on the test data and the `out/submission.csv` file will be generated.


## **Results**

We achieved a F1-score of 0.898 and the accuracy of 0.947 on the website using the unet model. All the details are in the report.
