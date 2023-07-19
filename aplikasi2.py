import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

title = 'Predict Inflation per Month in Indonesia 💸💸💸'
subtitle = 'Predict Inflation per Month in Indonesia Using Machine Learning💻'

def main():
    st.set_page_config(layout="centered", page_icon='💸💸💸', page_title='Let\'s Predict Inflation per Month in Indonesia!')
    st.title(title)
    st.write(subtitle)

    form = st.form("Data Input")
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        data = {
            'Tanggal Referensi': pd.date_range(start=start_date, end=end_date).to_list()
        }
        data = pd.DataFrame(data)

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        data['Tanggal Referensi'] = (pd.to_datetime(data['Tanggal Referensi']) - pd.to_datetime('2003-01-01')).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction using the loaded model
        predictions = model.predict(data)

        # Create a DataFrame to store the results
        results = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d'), 'Predicted Inflation (in percent)': predictions})

        # Format the predicted inflation values as integers
        results['Predicted Inflation (in percent)'] = results['Predicted Inflation (in percent)'].astype(int)

        # Visualize the results using matplotlib
        plt.style.use('dark_background') 
        plt.plot(results['Date'], results['Predicted Inflation (in percent)'])
        plt.xlabel('Year')
        plt.ylabel('Predicted Inflation (in percent)')
        plt.xticks(rotation=45)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title('Predicted Inflation in Indonesia over Time')
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of x-axis ticks
        
        st.pyplot(plt)

        # Format the Date column in the results DataFrame
        results['Date'] = pd.to_datetime(results['Date']).dt.strftime('%d-%m-%Y')

        # Optionally, you can also show the raw data in a table
        st.dataframe(results)

    st.write("For more information about this project, check here: [GitHub Repo](https://github.com/PrastyaSusanto/Inflation-In-Indonesia)")

if __name__ == '__main__':
    main()
