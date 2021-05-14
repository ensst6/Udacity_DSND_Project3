import sys
import pandas as pd
import numpy as np
import sqlite3
import fasttext
import re


def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from two .csv files with a common 'id' column into a dataframe.
    These should be the disaster-response messages and categories.

    Args:
        messages_filepath (string): path to the disaster messages file, usually 'disaster_messages.csv'
        categories_filepath (string): path to the message categories file, usually 'disaster_categories.csv'

    Returns:
        df: a pandas dataframe containing the messages & categories joined on the 'id' field; or
            any empty df if the either or both file paths are invalid.
    '''

    file_err = False
# load messages
    try:
        messages = pd.read_csv(messages_filepath)
    except:
        print('Messages file {} not found'.format(messages_filepath))
        file_err = True

# load categories
    try:
        categories = pd.read_csv(categories_filepath)
    except:
        print('Categories file {} not found'.format(categories_filepath))
        file_err = True

    if file_err:
        return pd.DataFrame()
    else:
# create a merged dataframe from the two files
        df = messages.merge(categories, on='id')
        return df


def clean_data(df, ft_path):
    '''
    Cleans the disaster-response data based on deficiencies identified in exploratory analysis.

    Args:
        df (pandas dataframe): dataframe with the merged messages & categories data
        ft_path (string): path to the fastText model 'lid.176.bin' for language identification

    Returns:
        df: a pandas dataframe containing the cleaned data; or
            an empty df if the fastText model file path is invalid.
    '''

# check the fastText path first; abort if wrong
    try:
        ft_model=fasttext.load_model(ft_path)
    except:
        print('fastText model file {} not found'.format(ft_path))
        return pd.DataFrame()

## CLEAN THE CATEGORIES ##
# split the categories field & create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

# get the 1st row and use it to make a list of category column names
    row = categories.iloc[0,:]
    category_colnames = row.str.rstrip('-01')
# rename the columns
    categories.columns = category_colnames

# get rid of the text and just keep the integer code
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories.loc[:,column].str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

# now merge the category columns with the main df
# drop the original categories column from `df` (no longer needed)
    df.drop(columns=['categories'], inplace=True)
# concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

## DROP DUPLICATE ROWS ##
    df.drop_duplicates(inplace=True, ignore_index=True)

## DROP THE RELATED = 2 ENTRIES SINCE IT'S UNCLEAR WHAT THEY REPRESENT (SEE EDA) ##
    df = df[~(df['related']==2)]

## DROP THE child_alone CATEGORY SINCE NO MESSAGES WERE IDENTIFIED AS MEMBERS OF IT ##
    df.drop(columns=['child_alone'], inplace=True)

## USE THE FASTTEXT MODEL TO IDENTIFY NON-ENGLISH MESSAGES, THEN REMOVE THEM ##
# note this gives a tuple, tho b/c it said "object" I assumed it was a string, til the replace stuff failed
# so, coercing to string. the 1st [0] subscript drops the probability element, the 2nd [0] extracts the label text
    df['pred_lang']=df['message'].apply(ft_model.predict).str[0].str[0]
# clean up the labeling
    df['pred_lang'] = df['pred_lang'].str.replace("__label__",'',regex=False)
# keep only the english ('en') messages
    df = df[df['pred_lang']=='en']
# 'pred_lang' is no longer needed, so drop it
    df.drop(columns=['pred_lang'], inplace=True)

## DROP THE ENTRIES THAT HAVE DUPLICATE ID VALUES & MESSAGES; KEEPING THE LAST ENTRY (SEE EDA) ##
    df.drop_duplicates(subset='id', inplace=True, ignore_index=True, keep='last')

    return df


def save_data(df, database_filename):
    '''
    Saves the cleaned disaster-response data into an SQLite database for modeling/analysis.

    Args:
        df (pandas dataframe): dataframe containing the cleaned data
        database_filename (string): path to the SQLite '.db' file (created if doesn't exist)

    Returns:
        None
    '''

# create SQLite connection; will create file if doesn't exist
    conn = sqlite3.connect(database_filename)

# get a cursor
    cur = conn.cursor()

# drop the messages table in case it already exists
    cur.execute("DROP TABLE IF EXISTS messages")

# create the table with its columns and 'id' as primary key
    sql_start = "CREATE TABLE messages (id INTEGER PRIMARY KEY , message TEXT, original TEXT, genre TEXT, "
    cols = df.columns[4:].to_list()
    all_cols = ' INTEGER, '.join(cols)
    sql_create = sql_start + all_cols + ' INTEGER);'
    cur.execute(sql_create)
# write the db file; commit & close the connection
    df.to_sql('messages', conn, index=False, if_exists='append')
    conn.commit()
    conn.close()

    return


def main():
    '''
    Pre-process data for the disaster-response message identification app.
    Expects to be passed a list of 5 file paths for the following:
        -- 'disaster_messages.csv' : the messages
        -- 'disaster_categories.csv': a list of categories for the messages
        -- a filename ending in '.db' for an SQLite database that will contain the cleaned data
           The file will be created if doesn't exist. If it exists, its 'messages' table will
           be dropped and replaced with the data from this script.
        -- 'lid.176.bin': the fastText language identification model.
            This is needed to identify non-English messages in the cleaning step.
            Note this algorithm hasn't been tested with the compact 'lid.176.ftz' model

    Example:
        > python3 etl_pipeline.py disaster_messages.csv disaster_categories.csv DisResp.db /tmp/lid.176.bin
    '''

    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath,\
        ft_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        if df.empty:
            print('Please try again with correct data file paths')
            sys.exit(1)

        print('Cleaning data...\n   FASTTEXT MODEL PATH: {}'
              .format(ft_path))

        df = clean_data(df, ft_path)
        if df.empty:
            print('Please try again with correct fastText file path')
            sys.exit(1)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively.\n '\
              'The third argument should be the filepath of the database to save the cleaned data.\n '\
              'Finally, provide a path to the fastText model file lid.176.bin. \n'\
              'Example: python3 etl_pipeline.py disaster_messages.csv disaster_categories.csv '\
              'DisResp.db /tmp/lid.176.bin')


if __name__ == '__main__':
    main()
