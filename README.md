# Disaster-Response-Pipeline
 
 In this project, I analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

In the data folder, you'll find a data set provided by [Figure Eight](https://www.figure-eight.com/) containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that one can send the messages to an appropriate disaster relief agency.

The project includes a [web app](https://my-app-disaster.herokuapp.com) where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

# project motivation :

* An ETL pipeline, that Extracts disaster-related messages classified in 36 categories (listed below), Transforms the data through a wrangling process so that they can be used to train a Machine Learning model, and saves (Loads) the data into a database.
* A Machine Learning pipeline. In this part the data is supplied to a model to be trained using natural language process techniques. The trained model is saved in a file that will be used to classify new messages.
* A web application in which a user can enter messages, which will be classified according to the mentioned categories.

# Project components

1. ETL Pipeline

The Python script, process_data.py, contains a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline

The Python script, train_classifier.py, has a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App

- It displays results in a Flask web app which is deployed on Heroku.

# Installation

run: pip install -r requirements.txt

$ pip install plotly

or

$ conda install -c plotly plotly

# Licence
Feel free to use the code and let me know if the model can be improved. If you would like to do further analysis, it is available below under a [Creative Commons CC0 1.0 Universal (CC0 1.0)](https://creativecommons.org/publicdomain/zero/1.0/) license.
 
