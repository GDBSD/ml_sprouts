# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st

"""Some functions to keep the main UI page clean"""

# Cache the data loaded so this function doesn't run every time the page
# is refreshed. When you mark a function with Streamlit’s cache annotation,
# it tells Streamlit that whenever the function is called that it should check
# three things:
# - The actual bytecode that makes up the body of the function
# - Code, variables, and files that the function depends on
# - The input parameters that you called the function with
@st.cache
def load_data(data_url, nrows, date_column):
    """Downloads some date, puts it in a Pandas dataframe, and converts the
    Pandas date column from text to datetime
    :param data_url: string - location of the data
    :param nrows: int - number of rows we want to load into the dataframe.
    :param date_column: string - converts the Pandas date column from text to datetime.
    :return: Pandas dataframe
    """

    data = pd.read_csv(data_url, nrows=nrows)

    def lowercase(x): return str(x).lower()

    data.rename(lowercase, axis='columns', inplace=True)
    data[date_column] = pd.to_datetime(data[date_column])
    return data
