# micrometeorite-ai

## Before you begin

Before you begin please make sure that all required libraries listed in `requirements.txt` are available.

Copy or rename `modules/config.template.py` to `modules/config.py` and set up everything you need.

## Predicting Micrometeorites

To predict pictures of possible micrometeorites you can use `ModelPredict.ipynb`. Pretrained Models for this task can be found HERE (Link to be added). Please keep in mind that these models have been trained with images that have been created using focus stacking. In addition these photos have been preprocessed.

The script lets you choose whether you want your pictures to be preprocessed or not by setting the variable `preprocessing` to `True` or `False`. Preprocessing the pictures means cutting out the biggest contour on the image, replacing the background with white color and creating a 1:1-format picture of the object without stretching or compressing it. To use this functionality, you’ll have to have a contour model. You can download one [here](https://github.com/opencv/opencv_extra/blob/5e3a56880fb115e757855f8e01e744c154791144/testdata/cv/ximgproc/model.yml.gz).

Needed `config.py`:

- TRAINED_MODEL
- PREDICT_DATA
- PREDICT_RESULTS (optional)

## Training Models

You can train your own models using `ModelTraining.ipynb` or `Hyperparameter_BayesianOptimization.ipynb` if you also want to do some Hyperparameteroptimization during the training process.

This Repository doesn't have any training data to share. If you only want to play around with ML you can use a data set provided by Microsoft which features pictures of cats and dogs. You can download it [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

Needed `config.py`:

- TRAINING_DATA
- MODEL_PATH
- IMG_SIZE
- TRAINING_EPOCHS
- PERCENT_TEST
