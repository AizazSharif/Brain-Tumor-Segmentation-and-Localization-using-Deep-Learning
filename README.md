# Brain-Tumor-Segmentation-and-Localization-using-Deep-Learning


This is an implementation of BRATS 2015 dataset for the purpose of Brain tumor segmentation and localization. It involves Flair, T1, T1c and T2 modalities with 4 type of tumors as Ground Truth.


## Installation

Clone the GitHub repository and install the dependencies.
* Install 
  * Anaconda (for creating and activating a separate environment)
  * keras-gpu=2.1.4=py35_0
  * numpy=1.13.3
  * matplotlib
  * tensorflow-gpu=1.0.1=py35_4
  * scikit-learn==0.19.1
  * SimpleITK
  * Skimage

* Clone the repo and go to the directory 
```
$ git clone https://github.com/AizazSharif/Brain-Tumor-Segmentation-and-Localization-using-Deep-Learning.git
$ cd Brain-Tumor-Segmentation-and-Localization-using-Deep-Learning

```

## Dataset

Dataset can be downloaded by making account on http://braintumorsegmentation.org/. Use the dataset with the data_prep.py paths accordingly.

For data prepration run :
```
python data_prep.py

```

## Training
You can train your own model by changing the setting in model_and_training.py. 

For training the model use :
```
python model_and_training.py

```

## Validation

Validation is done within model_and_training.py during the training. Testing and localization will soon be uploaded in a separate python script.






