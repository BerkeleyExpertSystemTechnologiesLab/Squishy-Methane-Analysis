import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def eval(predictions, truth_labels):
    accuracy = accuracy_score(truth_labels, predictions)
    f1 = f1_score(truth_labels, predictions)
    print('Overall Accuracy:', accuracy)
    print('F1 Score:', f1)
    conf_matrix = confusion_matrix(truth_labels, predictions)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for label, acc in zip(['Leak', 'Nonleak'], per_class_accuracy):
        print(f"Class '{label}' Accuracy: {acc:.4f}")

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Load a prediction file.")

    # Add an argument for the prediction file name
    parser.add_argument("file_name", help="Name of the prediction file")

    # Parse the command-line arguments
    args = parser.parse_args()

    predictions_df = pd.read_csv(args.file_name)

    predictions = predictions_df['Prediction']
    truth_labels = predictions_df['Label']

    eval(predictions, truth_labels)

if __name__ == "__main__":
    main()