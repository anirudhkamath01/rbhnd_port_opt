import streamlit as st
import pandas as pd
import robin_stocks.robinhood as robin

User Login Information
login = robin.login("anirudhkamath@berkeley.edu", "unirudh555")
totp = pyotp.TOTP("73JD3T6UFTJGJPRS").now()

#Pulls user portfolio from RH account
my_stocks = robin.build_holdings()
my_profile = robin.build_user_profile()

st.text(my_stocks)