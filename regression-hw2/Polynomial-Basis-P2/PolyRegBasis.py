# HW2 - Regression (Linear + Polynomial)
# Part 2: Polynomial Regression w/ Basis Expansion
# Jackson Chumbler
# CS 460G Spring 2022
# !!Preprocessing -> Added x^1,y header for synthetic datasets!!

import matplotlib.colors as colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("./data/synthetic-1.csv")
    name1 = "synthetic-1"
    df2 = pd.read_csv("./data/synthetic-2.csv")
    name2 = "synthetic-2"

    print(name1)
    run(df, 0.001, 1000, 5, name1)
    print(name2)
    # run(df2, 0.001, 1000, 5, name2)
    run(df2, 0.0001, 20000, 7, name2)


def run(df, learning, epochs, degree, name):
    basis_expansion(df, degree)
    coeffs = linear_gradient_descent(df, learning, epochs, degree)
    poly = coeffs_to_poly(coeffs)
    avg, mse = test_polynomial(df, poly)
    print("Polynomial:\n", poly)
    print("Avg: ", avg)
    print("MSE: ", mse)
    plot(df, poly, degree, name)

# Convert tuple coeffs to a list of polynomial coeffs


def coeffs_to_poly(mat):
    poly = []
    poly.append(mat[-1][1])
    for x in mat:
        if x[0] != "y" and x[0] != 'b0':
            poly.append(x[1])
    return poly

# Give y_hat given a polynomial


def predict_y(x, poly):
    y_hat = poly[0]
    for i in range(1, len(poly)):
        y_hat += poly[i]*(x**i)
    return y_hat

# Given a polynomial, test it on all rows,
# and return the avg-distance and M.S.E.


def test_polynomial(df: pd.DataFrame, poly):

    sum_of_difference = 0
    mean_squared_error = 0

    for i in range(df["y"].values.size):
        actual_y = df["y"].values[i]
        y_hat = predict_y(df["x^1"].values[i], poly)

        sum_of_difference += (y_hat - actual_y)
        mean_squared_error += (y_hat - actual_y)**2
    mean_squared_error /= df["y"].values.size
    avg_dist = sum_of_difference / df["y"].values.size

    return avg_dist, mean_squared_error

# Given a coeffs tuples list, test it on all rows,
# and return the avg-distance and M.S.E.


def test_prediction(df: pd.DataFrame, coeffs):
    sum_of_difference = 0
    mean_squared_error = 0
    for i in range(df["y"].values.size):
        true_qual = df["y"].values[i]
        estimated_qual = 0  # calculate using all labels and coeffs
        for coef in coeffs:
            if coef[0] != 'b0':
                estimated_qual += coef[1] * df[coef[0]].values[i]
            else:
                estimated_qual += coef[1]
        sum_of_difference += (estimated_qual - true_qual)
        mean_squared_error += (estimated_qual - true_qual)**2
    mean_squared_error /= df["y"].values.size
    avg_distance = sum_of_difference / df["y"].values.size
    return avg_distance, mean_squared_error

# Create polynomial coefficients array which
# preprocesses data to find reasonable starting
# values.


def basis_expansion(df: pd.DataFrame, degree):
    x_vals = list(df["x^1"])
    length = len(x_vals)
    mat = np.zeros((length, degree))
    for i in range(1, degree):
        tmp_arr = []
        for j in range(length):
            mat[j, i] = x_vals[j]**(i+1)
            tmp_arr.append(x_vals[j]**(i+1))
        label = "x^" + str(i+1)
        df[label] = tmp_arr
    return mat

# This is just an adaptation of the same function in part one,
# using different names. The partial derivative is calculated
# implicitly by including it within another for-loop. This
# allows greater control on which coefficient it is applied,
# as well as allowing the weights to change per row.


def linear_gradient_descent(df: pd.DataFrame, learning_rate: float, epochs: int, degree: int):
    # Fill coeffs arr with correct names
    coeffs = []
    for i in range(1, degree + 1):
        name = "x^" + str(i)
        coeffs.append([name, 0])
    offset, blah = test_prediction(df, coeffs)
    coeffs.append(["b0", offset])

    for epoch in range(epochs):
        for row in range(df["y"].values.size):
            actual_quality = df["y"].values[row]
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

# Graphs the function and a scatterplot of points.
# The points are color in accordance to their distance from
# The prediction function.


def plot(df, poly, deg, name: str):

    x = df["x^1"].values
    y = df["y"].values
    min_dist = float('inf')
    max_dist = 0
    y_predict_l = []
    x1 = np.sort(x)
    color_arr = []
    for x_val in x1:
        yhat = poly[0]
        for i in range(1, len(poly)):
            yhat += poly[i]*(x_val**i)

        y_act = 0
        y_predict_l.append(yhat)
        for i in range(df["x^1"].values.size):
            if df["x^1"].values[i] == x_val:
                y_act = df["y"].values[i]
                break
        diff = y_act - yhat

        color_arr.append(abs(diff))
        if abs(diff) < min_dist:
            min_dist = abs(diff)
        elif abs(diff) > max_dist:
            max_dist = abs(diff)

    norm = colors.Normalize(vmin=min_dist, vmax=max_dist, clip=True)
    from matplotlib import cm
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("cool"))
    for c in range(len(color_arr)):
        color_arr[c] = mapper.to_rgba(color_arr[c])
    fig, ax = plt.subplots()
    ax.plot(x1, y_predict_l, color="purple")

    df = df.sort_values(by='x^1')
    x = df["x^1"].values
    y = df["y"].values
    ax.scatter(x, y, color=color_arr)

    ax.set_xlabel("x values")
    ax.set_ylabel("y values")
    title = name + ", deg=" + str(deg)
    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    main()
