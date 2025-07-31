import streamlit as st
import pandas as pd

@st.cache_data
def fx(fname):
    df = pd.read_csv(fname)
    return df

st.title('emp sales data')

fname = st.text_input('Enter csv file:')

#col = st.text_input("Filter column:")

if fname:
    df = fx(fname)
    st.write("Loaded data")
    st.write(df)
    
