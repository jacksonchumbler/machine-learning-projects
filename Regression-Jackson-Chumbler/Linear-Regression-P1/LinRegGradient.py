# HW2 - Regression (Linear + Polynomial)
# Part 1: Linear Regression w/ Gradient Descent
# Jackson Chumbler
# CS 460G Spring 2022



import pandas as pd
import numpy as np
import time

def main():
    df = pd.read_csv("./data/winequality-red.csv")

    print("Initial Coefficients:\n")
    coeffs = init_coefficients(df)
    avg_error, mse = test_prediction(df, coeffs)
    for c in coeffs:
        print(c)
    print("\nAverage Error: ", avg_error)
    print("Mean Squared Error: ", mse)
    print("------------------------------------------")
    print("\nImproved by Gradient Descent:\n")
    start_time = time.time()
    coeffs = linear_gradient_descent(df, 0.0001, 100)
    end_time = time.time()
    avg_error, mse = test_prediction(df, coeffs)
    for c in coeffs:
        print(c)
    print("\nAverage Error: ", avg_error)
    print("Mean Squared Error: ", mse)
    print(end_time - start_time)

# Estimate the coefficient of each
# independent variable. This is done
# by taking finding the slope using all points
# at the highest and lowest quality.


def init_coefficients(df: pd.DataFrame):
    label_est_coefficient = []
    max_qual = max(df["quality"])
    min_qual = min(df["quality"])
    indices_at_max_quality = df.loc[df["quality"] == max_qual].index
    indices_at_min_quality = df.loc[df["quality"] == min_qual].index
    sum_max = 0
    sum_min = 0
    for label, values in df.iteritems():
        if label != "quality":
            for i in indices_at_max_quality:
                sum_max += values[i]
            for i in indices_at_min_quality:
                sum_min += values[i]
            avg_max = sum_max / indices_at_max_quality.size
            avg_min = sum_min / indices_at_max_quality.size
            estimated_coefficient = (max_qual - min_qual) / (avg_max - avg_min)
            estimated_coefficient /= df.columns.size - 1
            label_est_coefficient.append([label, estimated_coefficient])
    offset, mse = test_prediction(df, label_est_coefficient)
    label_est_coefficient.append(["b0", offset])
    return label_est_coefficient


def test_prediction(df: pd.DataFrame, coeffs):
    sum_of_difference = 0
    mean_squared_error = 0
    for i in range(df["quality"].values.size):
        true_qual = df["quality"].values[i]
        estimated_qual = 0  # calculate using all labels and coeffs
        for coef in coeffs:
            if coef[0] != 'b0':
                estimated_qual += coef[1] * df[coef[0]].values[i]
            else:
                estimated_qual += coef[1]  # offset
        sum_of_difference += (estimated_qual - true_qual)
        mean_squared_error += (estimated_qual - true_qual)**2
    mean_squared_error /= df["quality"].values.size
    avg_distance = sum_of_difference / df["quality"].values.size
    return avg_distance, mean_squared_error


def linear_gradient_descent(df: pd.DataFrame, learning_rate: float, epochs: int):
    coeffs = init_coefficients(df)  # Preproccessed, "good" starting coeffs...
    for epoch in range(epochs):
        for row in range(df["quality"].values.size):
            actual_quality = df["quality"].values[row]
            est_quality = 0
            for coef in coeffs:
                if coef[0] == 'b0':
                    est_quality += coef[1]
                else:
                    est_quality += coef[1] * df[coef[0]].values[row]
            difference = (est_quality - actual_quality)
            for coef in coeffs:
                if coef[0] != 'b0':
                    coef[1] -= learning_rate*difference*df[coef[0]].values[row]
                else:
                    coef[1] -= learning_rate*difference
    return coeffs


if __name__ == "__main__":
    main()
