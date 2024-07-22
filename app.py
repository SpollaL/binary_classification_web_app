
from typing import Tuple, Literal
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from settings import classifiers_settings, strtofun


def main():
    st.title("Binary Classification web app")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data(persist=True)
    def load_data() -> pd.DataFrame:
        """load dataset
        """
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """_summary_

        Args:
            df (pd.DataFrame): input data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: train test spilt dataframes
        """
        y = df.type
        x = df.drop(columns={"type"})
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(
        metrics_list: Literal["Confusion Matrix", "ROC curve", "Precision Recall curve"]
        ):
        """plot metrics

        Args:
            metrics_list (Literal[Confusion Matrix; ROC curve, Precision Recall curve&quot]):
                list of metrics to plot
        """
        for metric in metrics_list:
            st.subheader(metric)
            fig = strtofun[metric].from_estimator(model, x_test, y_test)
            st.pyplot(fig.figure_)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Ser (Classification)")
        st.write(df)

    st.sidebar.subheader("Choose Classifier")

    classifier = st.sidebar.selectbox(
        "Classifier",
        tuple(classifiers_settings.keys())
        )

    st.sidebar.subheader("Model hyperparameters")
    widgets = {}
    for param_name, param_params in classifiers_settings[classifier]["params"].items():
        widgets[param_name] = param_params["fun"](**param_params["params"])

    metrics = st.sidebar.multiselect("what metrics to plot?", tuple(strtofun.keys()))

    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier)
        model = classifiers_settings[classifier]["model"](**widgets)
        model.fit(x_train, y_train)
        test_accuracy = model.score(x_test, y_test)
        y_pred_test = model.predict(x_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        st.write("Accuracy: ", round(test_accuracy, 2))
        st.write("Precision: ", round(test_recall, 2))
        st.write("Precision: ", round(test_precision, 2))
        plot_metrics(metrics)


if __name__ == "__main__":
    main()
