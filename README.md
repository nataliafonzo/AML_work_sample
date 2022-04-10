# AML work sample

This repository contains the files needed to train and evaluate the performance of a binary classifier (new or used) for items in `MLA_100k.jsonlines`. 


## Files

- `new_or_used.py` imports the necessary classes and functions to extract features from `MLA_100k.jsonlines` and then train the classifier. At the end, it **displays the performance** obtained in the test set. 
- `feature_extraction.py` defines the function that **extracts features** from the json file.
- `dummies_transformer.py`, `fillnan_transformer.py`, `features_selector.py`  are found in the `transformers` folder. These files define the classes involved in the pipeline's **pre-processing steps**.
- `settings.py` contains the model's **hyperparameters** and auxiliary lists of columns/features involved in the pre-processing and training steps.
- `classifier.py` **defines the model's class** and a function to **train** it.

An `environment.yml` file is also provided to create a conda environment with all required packages installed.


## Steps to run the code

First, clone this repository or manually download the above-mentioned files:
```
$ git clone git@github.com:nataliafonzo/AML_work_sample.git
```
Make sure you save `MLA_100k.jsonlines` in the just-cloned AML_work_sample folder. Then change the working directory to:
```
$ cd ./AML_work_sample
```
Create a conda environment as follows: 
```
$ conda env create -f environment.yml
``` 
This will also install the required Python packages. Activate the environment:
```
$ conda activate AML_work_sample
```
Finally, train and evaluate the model: 
```
$ python new_or_used.py
```
