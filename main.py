import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Dense,Dropout
# from tensorflow.keras.models import Sequential 
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_wine, load_digits, load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

import streamlit as st
import pandas as pd
import altair as alt




hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(  hide_st_style, unsafe_allow_html=True)
# Add Image to streamlit sidebar
st.sidebar.image("NeuroWorksTextlogo.png", width=200)
# Title
st.title("NeuroWorks")


# Choose which function to perform in streamlit
parts_options = ['Select','Models', 'Neural', 'Custom']
parts_choice = st.selectbox("Select",parts_options)

if parts_choice == "Models":
        # Dataset
    dataset_options = ["Iris", 'Diabetes', 'Wine']
    dataset_choice = st.sidebar.selectbox("Select dataset", dataset_options)

    if dataset_choice == "Iris":
        iris = load_iris()
        X = iris.data
        y = iris.target


    elif dataset_choice == "Wine":
        wine = load_wine()
        X = wine.data
        y = wine.target

    elif dataset_choice == "Diabetes":
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target


    # Train/Test Split

    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    random_state = st.sidebar.slider("Random state", 1, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    model_options = {
        "K Nearest Neighbors": KNeighborsRegressor,
        "Decision Tree": DecisionTreeRegressor,
        "Random Forest": RandomForestRegressor,

    }

    model_choice = st.sidebar.selectbox("Select model", list(model_options.keys()))

    model = model_options[model_choice]()
    # Hyperparameters
    if model_choice == "K Nearest Neighbors":
        n_neighbors = st.sidebar.slider("Number of neighbors", 1, 15, 5)
        model.n_neighbors = n_neighbors

    elif model_choice == "Decision Tree":
        max_depth = st.sidebar.slider("Maximum depth", 1, 15, 5)
        model.max_depth = max_depth


    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)


    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Accuracy score
    acc = model.score(X_test, y_test)

    st.write(f"Model: {model_choice}")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Accuracy: {acc}")

    # Visualization
    fig = px.scatter(x=X_test[:, 0], y=X_test[:, 1], color=y_pred)
    st.plotly_chart(fig)


elif parts_choice == "Neural":
# Define datasets
    dataset_options = {
        "Wine": load_wine(),
        "MNIST Digits": load_digits(),
        "Iris": load_iris(),
        "Breast Cancer": load_breast_cancer(),
        "Diabetes": load_diabetes()
    }

    # Select dataset
    dataset_name = st.sidebar.selectbox("Choose a dataset", list(dataset_options.keys()))
    dataset = dataset_options[dataset_name]

    X = dataset.data
    y = dataset.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
    epochs = st.sidebar.slider("Epochs", 10, 100, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    hidden_layers = st.sidebar.slider("Hidden Layers", 1, 4, 2)

    # Define activation function options
    activation_functions = ["relu", "sigmoid", "tanh"]
    activation_func = st.sidebar.selectbox("Activation Function", activation_functions)

    # Define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(X_train.shape[1], activation=activation_func, input_shape=(X_train.shape[1],)))

    # Add hidden layers
    for i in range(hidden_layers):
        model.add(tf.keras.layers.Dense(10, activation=activation_func))

    model.add(tf.keras.layers.Dense(1, activation="linear"))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])

    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Display results

    st.text(f"Dataset: {dataset_name}")
    st.write(f"Hyperparameters: Learning Rate={learning_rate}")
    st.write(f"Epochs={epochs} ")
    st.write(f"Batch Size={batch_size}")
    st.write(f"Hidden Layers={hidden_layers}")
    st.write(f"Activation Function={activation_func}")


    st.text(f"Accuracy: {mae}")


elif parts_choice == "Custom":

    st.title("Custom Dataset Visualization")

    # Upload file
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # If user uploaded file
    if uploaded_file is not None:
        # Load data into dataframe
        df = pd.read_csv(uploaded_file)
        # Display dataframe
        st.write(df)
        
        # Choose columns for scatter plot
        x_column = st.sidebar.selectbox("Choose a column for the x-axis", options=list(df.columns))
        y_column = st.sidebar.selectbox("Choose a column for the y-axis", options=list(df.columns))
        color_column = st.sidebar.selectbox("Choose a column for the color encoding", options=list(df.columns))
        
        # Create scatter plot using Altair
        chart = alt.Chart(df).mark_circle().encode(
            x=alt.X(x_column, type='quantitative'),
            y=alt.Y(y_column, type='quantitative'),
            color=color_column,
            tooltip=[x_column, y_column, color_column]
        ).interactive()
        
        # Display scatter plot
        st.altair_chart(chart, use_container_width=True)
    
    else:
        st.write("Please upload a CSV file to get started.")

elif parts_choice == "Select":
    st.write("Select the option you want to see")
    