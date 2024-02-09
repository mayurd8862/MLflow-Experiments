import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the bank churn prediction dataset
    try:
        data = pd.read_csv('Churn_Modelling.csv')
    except Exception as e:
        logger.exception(
            "Unable to load dataset. Error: %s", e
        )
    X = data.drop(columns=['CustomerId','RowNumber','Surname','Geography','Gender'],axis=1)
    y = data['Exited']

    test_size =0.15
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=test_size,random_state=42)

    # Train a logistic regression model
    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(train_x, train_y)

        predicted_labels = model.predict(test_x)

        # Evaluate model metrics
        accuracy, precision, recall, f1 = eval_metrics(test_y, predicted_labels)

        print("Logistic Regression Model:")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Log parameters and metrics to MLflow
        mlflow.log_params({
            "model_type": "Logistic Regression",
            "test size" : test_size
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Log the trained model
        mlflow.sklearn.log_model(model, "model")
