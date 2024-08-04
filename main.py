import numpy as np
import pickle
import pandas as pd
import streamlit as st


pickle_dtree = open("yield_pred_dtree.pkl", "rb")
pickle_ran = open("yield_pred_rand.pkl", "rb")
pickle_GB = open("yield_pred_GB.pkl", "rb")
pickle_knn = open("yield_pred_knn.pkl", "rb")
pickle_svm = open("yield_pred_svr.pkl", "rb")

classifier = pickle.load(pickle_dtree)
classifier1 = pickle.load(pickle_ran)
classifier2 = pickle.load(pickle_GB)
classifier3 = pickle.load(pickle_knn)
classifier4 = pickle.load(pickle_svm)

st.header("MACHINE LEARNING BASED PREDICTION OF CLIMATE EVENT RISKS TO MAIZE YIELD")
st.text("This project is about predicting maize yield based on climate factors")


def main():
    st.title("Maize Yield Prediction")
    html_temp = """
    <div style="background-color:teal; padding:10px;">
    <h2 style="color:white; text-align:center;">Make Your Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Decision Tree Regression', 'Random Forest Regression', 'Gradient Boosting Regression',
                  'K-Nearest Neighbor(KNN)', 'Support Vector Machine(SVM)']
    option = st.sidebar.selectbox(
        'which model would you like to use ?', activities)
    st.subheader(option)
    pesticide = st.text_input('Input your pesticide level here:', 'Type Here')
    rainfall = st.text_input(
        'Input your average rainfall here:', 'Type Here')
    temperature = st.text_input(
        'Input your temperature here:', 'Type Here')

    inputs = [[temperature, rainfall, pesticide]]

    result = ""

    if st.button('Predict'):
        if option == 'Random Forest Classification':
            result = classifier1.predict(inputs)[0]
            st.success('Prediction Result : {}'.format(result))
        elif option == 'K-Nearest Neighbor(KNN)':
            result = classifier2.predict(inputs)[0]
            st.success('Prediction Result : {}'.format(result))
        elif option == 'Support Vector Machine(SVM)':
            result = classifier3.predict(inputs)[0]
            st.success('Prediction Result : {}'.format(result))
        else:
            result = classifier.predict(inputs)[0]
            st.success('Prediction Result : {}'.format(result))

        # result=predict_note_authentication(temperature,rainfall,pesticide)
        # st.success('the output is {}'.format(result))

    if st.button('about'):
        st.text("This project is about predicting maize yield based on climate factors")


@st.cache_data
def load_data(nrows):
    data = pd.read_csv('main_df.csv', nrows=nrows)
    return data


data_list = load_data(1000)

st.subheader('MAIZE DATA')
st.write(data_list)


if __name__ == '__main__':
    main()
