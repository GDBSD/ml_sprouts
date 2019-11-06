# -*- coding: utf-8 -*-

import os
import sys
import streamlit as st
import numpy as np
from PIL import Image

# Append the project path to sys.path so we can import modules
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from helpers import streamlit_helpers

# This is a demo of the Streamlit package using Uber data.
# To run the demo navigate to this file location with a terminal
# and execute the command: "streamlit run uber_demo.py"

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
IMAGE_PATH = '{}/media'.format(module_path)

# Create the header. We need to convert the png to an RGB object to open it.
image_path = '{}/uber.png'.format(IMAGE_PATH)
Image.open(image_path).convert('RGB')

st.image(image_path, caption='', use_column_width=False)
st.title('Impactful Data Science Demos')
st.subheader("Exploring Uber Data with < 50 Lines of Code")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data complete.')

# Load 10,000 rows of data into the dataframe.
data = streamlit_helpers.load_data(DATA_URL, 10000, DATE_COLUMN)

# Add a subheader and display the raw data.
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(data)

if st.checkbox('Show number of pickups by hour'):
    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(
        data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values)

# Use Streamlit st.map() function to overlay the data on a map of New York City.
if st.checkbox('Show on map'):
    st.subheader('Data Projected on NYC Map')
    st.write('Zoom or pan for more detail.')
    hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.map(filtered_data)



