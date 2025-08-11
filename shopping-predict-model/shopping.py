from typing import Counter
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

TEST_SIZE = 0.4

def plot_class_distribution(labels):
    counter = Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.bar(classes, counts, tick_label=['No Revenue (0)', 'Revenue (1)'])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.show()

def get_month(month_name):
    month_map = {'Jan': 0 ,'Feb': 1 , 'Mar': 2 ,'Apr': 3 ,'May': 4 ,'June': 5 , 'Jul': 6 , 'Aug': 7 , 'Sep': 8 ,'Oct': 9 ,'Nov': 10 , 'Dec': 11}
    return (month_map[month_name])

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    
    plot_class_distribution(labels)

    # Find the best estimator
    best_model = discover_best_estimator(X_train, y_train)

    # Train model (best_model is already fitted here, so we could skip retraining)
    model = train_model(X_train, y_train, best_model)
    predictions = model.predict(X_test)

    # Evaluate performance
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv(filename)

    result = ()

    evidences = []
    labels = []

    for _, row in df.iterrows():
        row['VisitorType'] = 1 if row['VisitorType'] == "Returning_Visitor" else 0
        row['Weekend'] = 1 if row['Weekend'] == True else 0
        row['Month'] = get_month(row['Month'])
        evidences.append([
            int(row['Administrative']),
            float(row['Administrative_Duration']),
            int(row['Informational']),
            float(row['Informational_Duration']),
            int(row['ProductRelated']),
            float(row['ProductRelated_Duration']),
            float(row['BounceRates']),
            float(row['ExitRates']),
            float(row['PageValues']),
            float(row['SpecialDay']),
            int(row['Month']),
            int(row['OperatingSystems']),
            int(row['Browser']),
            int(row['Region']),
            int(row['TrafficType']),
            int(row['VisitorType']),
            int(row['Weekend']),
        ])
        
        labels.append(1 if row['Revenue'] == True else 0)

    result = (evidences, labels)    
    return result


def train_model(evidence, labels, model):
    """
    Given evidence and labels, fit the provided model (pipeline) on the data.
    """
    model.fit(evidence, labels)
    return model


def discover_best_estimator(evidence, labels):
    """
    Find the best KNN configuration using GridSearchCV.
    Returns the trained pipeline with the best parameters.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1'
    )

    grid.fit(evidence, labels)

    print(f"Best k: {grid.best_params_['knn__n_neighbors']}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    true_positive_rate = 0
    true_negative_rate = 0
    for i, _ in enumerate(labels):
        if labels[i] == 1 and predictions[i] == 1:
            true_positive_rate += 1
        elif labels[i] == 0 and predictions[i] == 0:
            true_negative_rate += 1 
    
    sensitivity = true_positive_rate / list(labels).count(1)
    specificity =  true_negative_rate / list(labels).count(0)
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
