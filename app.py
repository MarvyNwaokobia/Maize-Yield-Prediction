import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def get_data(filename):
    maize_data = pd.read_csv(filename)

    return maize_data


with header:
    st.header(
        "MACHINE LEARNING BASED PREDICTION OF CLIMATE EVENT RISKS TO MAIZE YIELD")
    st.text("This project is about predicting maize yield based on climate factors")


with dataset:
    st.header('Climate and Maize Dataset')
    st.text('This dataset was gotten from kaggle')

    maize_data = get_data('main_df.csv')
    # st.write(maize_data.head())

    st.subheader('Maize yield distribution on the dataset')
    maize_yield = pd.DataFrame(
        maize_data['hg/ha_yield'].value_counts()).head(50)
    st.bar_chart(maize_yield)


with model_training:
    st.header('Time to train the model!')
    st.write('Here you get to choose the hyperparameters of the model and see how the performance changes!')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?',
                               min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[
                                     10, 20, 30, 'No limit'], index=0)

    input_feature = sel_col.text_input(
        'Which feature should be used as the input feature?', 'hg/ha_yield')

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(
            max_depth=max_depth, n_estimators=n_estimators)

    X = maize_data[[input_feature]]
    y = maize_data[['hg/ha_yield']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    y_true = [3, 5, 7, 9]
    y_pred = [2.8, 4.6, 7.2, 9.3]
    r2 = r2_score(y_true, y_pred)

    r2_percentage = r2 * 100

disp_col.subheader('R squared score of the model is:')
disp_col.write(r2_score(y, prediction))

# Display the R2 score with percentage
disp_col.subheader('R squared score with percentage:')
disp_col.write(f"R2 Score: {r2_percentage:.2f}%")
