# EXOPLANET HUNTING

Machine learning project to classify exoplanets using light fluctuation data from NASA's Kepler telescope. In the code section of the project you can find scripts for the different models. In the notebook section of the project you can see a visual walk through of the data analysis and model comparison of a NN and SVM.

This is a relatively difficult task due to the unbalanced data set, there are very few confirmed exoplanets in the data set, and many confirmed non-exoplanets.

# requirements

rMake sure you are using python3.6 (highest version of python tensorflow works with). Create a virtual environment by running:

```
python3.6 -m venv env
```

Once the environment is created run:

```
source env/bin/activate
```

Then go ahead an install the requirements:

```
python3.6 -m pip install --upgrade pip
python3.6 -m pip install -r requirements.txt
```

## running the program

If you want to try out the CLI demo, enter the code folder and run:

```
python3.6 demo.py
```

The demo program will either let you see the overall accuracy of each model, pay attention to the confusion matrix presented, the top left and bottom right values are the true positives, and the true negatives respectively. These numbers are more important than overall accuracy, due to if a model predicts everything as a non-exoplanet then it will get an overall high accuracy, which is false. 

If you ender `P/p` you can select an individual planet to predict, it will tell you the actual label, then each of the models predictions, then will show you the light fluctuation graph. exit out of the graph to continue the program.

If you want to try out the driver program, which is used to mass reset the models and save the models run:

```
python3.6 driver.py
```

# Data set

data can be found from Kaggle at the following link: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/downloads/kepler-labelled-time-series-data.zip

