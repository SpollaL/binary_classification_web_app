import streamlit as st
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

strtofun = {
        "Confusion Matrix": ConfusionMatrixDisplay,
        "ROC curve": RocCurveDisplay,
        "Precision Recall curve": PrecisionRecallDisplay,
    }

classifiers_settings = {
    "Support Vector Machine (SVM)": {
        "model": SVC, "params": {
            "C": {
                "fun": st.sidebar.number_input,
                "params": {
                    "label": "C (Regularization Parameter)", 
                    "min_value": 0.01,
                    "max_value": 10.0,
                    "step":0.01,
                    "key":"C"
                    },
            },
            "kernel": {
                 "fun": st.sidebar.radio,
                 "params":{
                       "label": "kernel",
                        "options": ("rbf", "linear"),
                         "key":"kernel"
                        },
                },
            "gamma": {
                 "fun": st.sidebar.radio,
                 "params":{
                       "label": "Gamma (Kernel Coefficient)",
                        "options": ("scale", "auto"),
                         "key":"gamma"
                        },
                },
            },
        },
    "Logistic Regression": {
        "model": LogisticRegression,
        "params": {
            "C": {
                "fun": st.sidebar.number_input,
                "params": {
                    "label": "C (Regularization Parameter)", 
                    "min_value": 0.01,
                    "max_value": 10.0,
                    "step":0.01,
                    "key":"C-LRD"
                    },
            },
            "max_iter": {
                "fun": st.sidebar.slider,
                "params": {
                    "label": "Maximum number of iterations",
                    "min_value": 100,
                    "max_value": 500,
                    "key": "max_iter",
                }
            },
        },
    },
    "Random Forest": {
            "model": RandomForestClassifier, "params": {
            "n_estimators": {
                "fun": st.sidebar.number_input,
                "params": {
                    "label": "The number of Trees in a forest", 
                    "min_value": 100,
                    "max_value": 5000,
                    "step":10,
                    "key":"n_estimators"
                    },
            },
            "max_depth": {
                "fun": st.sidebar.number_input,
                "params": {
                    "label": "Maximum Depth of a tree", 
                    "min_value": 1,
                    "max_value": 20,
                    "step":1,
                    "key":"max_depth"
                    },
            },
            "bootstrap": {
                 "fun": st.sidebar.radio,
                 "params":{
                       "label": "Bootstrap samples when building trees",
                        "options": (True, False),
                         "key":"bootstrap"
                        },
                },
        },
    },
}
