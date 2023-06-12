import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def regression(df, epochs_limit, learning_rate, x, y, intercept_priority):
    m = 0
    c = 0
    m_ar = []
    c_ar = []

    losses = []
    data_len = len(df)
    loss = 100000000
    loss_dif = epochs_limit + 1
    epoch_count = 0
    log = st.empty()
    while(loss_dif > epochs_limit):
        sum_m = 0
        sum_c = 0
        dm = 0
        dc = 0
        prev_loss = loss
        for d in range(data_len):
            sum_m = sum_m + (df[x][d] * (df[y][d] - (m*df[x][d]+c)))
            sum_c = sum_c + (df[y][d] - (m*df[x][d]+c))
            loss = loss + ((df[y][d] - (m*df[x][d]+c))
                           * (df[y][d] - (m*df[x][d]+c)))
        dm = (-2/data_len)*sum_m
        dc = (-2/data_len)*sum_c * intercept_priority
        loss = loss/data_len

        m = m-learning_rate*dm
        c = c-learning_rate*dc
        losses.append(loss)

        m_ar.append(m)
        c_ar.append(c)
        loss_dif = prev_loss - loss
        epoch_count = epoch_count+1

    return losses, m, c, epoch_count


def run(epochs_limit, intercept_priority):
    x = "x"
    y = "y"

    st.header("Dataset and Scatter Plot     ")
    df = pd.read_csv("dataset.csv", usecols=[x, y])
    col1, col2 = st.columns([1, 2])
    col1.dataframe(df)

    fig1 = plt.figure(1)
    plt.scatter(df[x], df[y])
    plt.title("Scatter Plot")
    col2.pyplot(fig1)

    losses, m, c, epochs = regression(
        df, epochs_limit, 0.0000001, x, y, intercept_priority)
    st.header("Predictions ")
    d1, d2 = st.columns(2)
    d1.metric(label="Slope ", value=m)
    d2. metric(label="Intecept ", value=c)


def prep_dataset(dataset):
    df = pd.read_csv(dataset)
    x = str(st.sidebar.selectbox("Select dependent data", df.columns))
    y = str(st.sidebar.selectbox("Select independent data", df.columns))

    if st.sidebar.button("Confirm"):
        x_data = df[x]
        y_data = df[y]
        dataset = np.column_stack([x_data, y_data])
        df = pd.DataFrame(dataset, columns=["x", "y"])
        df.to_csv("dataset.csv")

def main():
        with st.sidebar.subheader('Upload your dataset'):
            dataset = st.sidebar.file_uploader(
                "Please upload a file of type: csv", type=["csv"])
        if dataset is not None:
            prep_dataset(dataset)

        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        intercept_priority = float(st.sidebar.text_input(
            "Enter Intercept Priority parameter ", 1000))

        st.sidebar.write("***")
        epochs_limit = float(st.sidebar.text_input(
            "Enter Learning Rate Limit ", 0.001))

        if st.sidebar.button("Start Regression"):
            run(epochs_limit, intercept_priority)


if __name__ == "__main__":
    main()
