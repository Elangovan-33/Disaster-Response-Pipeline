# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Load the csv files and combine them into a data frame
    --
    Inputs:
        messages_filepath: csv file contains messages
        categories_filepath: csv file contains categories
    Outputs:
        df: the combined dataframe
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Load a dataframe and clean it
    --
    Inputs:
        df: dataframe

    Outputs:
        df: clean dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))

    # select the first row of the categories dataframe
    row = df.categories.str.split(';',expand=True).loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [category[:-2] for category in row ]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories.columns:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda name :name[-1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df.reset_index(drop=True), categories.reset_index(drop=True)], axis= 1)

   # drop duplicates
    df.drop_duplicates(inplace=True)

    # remove 2 from 'related' column
    df = df[(df['related'] == 1) | (df['related'] == 0)]

    return df


def save_data(df, database_filename):
    """
    Load a dataframe and save it to a database
    --
    Inputs:
        df: dataframe
        database_filename: database path
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, chunksize=1000, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
