"""
PLA.py
Author: Vincent Chao

This program models the perceptron learning algorithm as a linear
approximation against randomly generated data points. The points will be
displayed as a scatterplot, with the real and predicted target functions
plotted as lines over the data. The default target function is listed under
the global variables as a tuple (a, b) for the equation: y=a*x+b.
"""
import numpy as np
import matplotlib.pyplot as plt

"""
GLOBAL VARIABLES
"""
tf = (1, 3)  # Target function slope and y-int in equation of line: y=a*x+b
D = 200  # Number of points in data set
RANGE = 10  # Range of data to learn from; also limits the plot axes
MEAN_X = RANGE/2
MEAN_Y = tf[0] * MEAN_X + tf[1]


def f(x_n, g=tf):
    """
    Returns y-coor based on target function f and input x
    :param x_n: input
    :param g: Best-fit slope and y-offset for target function
    :return: y-coordinate based on target function f and input x
    """
    return g[0]*x_n+g[1]


def h(x_n, g=tf):
    """
    Takes input x and returns output +1 or -1 based on target function
    :param x_n: input data point
    :param g: Best-fit slope and y-offset for target function
    :return: output h(x), either +1 or -1
    """
    if x_n[2] > f(x_n[1], g):
        return 1
    else:
        return -1


def plot_target_function(graph, g=tf, color="k"):
    """
    Maps a line onto a plot to indicate ideal or estimated target function
    :param graph: The figure/plot to display the line on
    :param g: Best-guess target function parameters ([slope, y-int])
    :param color: Determines color of the target function line plot
    :return: none
    """
    plt.plot((0, RANGE), (f(0, g), f(RANGE, g)), color+":", figure=graph)


def plot(data, output, g=tf, show_g=True, show_tf=False):
    """
    Generate scatterplot using matplotlib.pyplot with a linear approximation
    of the best-guess target function
    :param data: Array of size Dx3; [0] is x-coor, [1] is y-coor, [2] is h(x)
    :param output: Array of size Dx1, containing y-output values for each point
    :param g: Best-guess target function parameters ([slope, y-int])
    :param show_g: Shows the predicted target function line
    :param show_tf: Shows the real target function line
    :return: none
    """
    fig = plt.figure()

    # Plot points
    for i, d in enumerate(data):
        if output[i] > 0:
            plt.scatter(d[1], d[2], figure=fig, marker=".", c="blue")
        else:
            plt.scatter(d[1], d[2], figure=fig, marker=".", c="red")

    # Show predicted target function line
    if show_g:
        plot_target_function(fig, g, color='g')

    # Show real target function line
    if show_tf:
        plot_target_function(fig, tf)

    # Plot properties
    plt.xticks(np.arange(int(min(data[:,1])), max(data[:,1]), 1.0))
    plt.yticks(np.arange(int(min(data[:,2])), max(data[:,2]), 1.0))
    plt.xlim(int(min(data[:,1]))-1, max(data[:,1])+1)
    plt.ylim(int(min(data[:,2]))-1, max(data[:,2])+1)
    plt.grid(True)  # turn on grid
    plt.xlabel('x-axis')  # x-axis label
    plt.ylabel('y-axis')  # y-axis label
    plt.title('Perceptron Learning Algorithm')  # title
    plt.show()


def perceptron_learning_algorithm(data, output):
    """
    Runs the PLA on a data set to find a linear classification
    :param data: Matrix of data points [[bias, x1, x2]]
    :param output: Array of y-output values, either +1 or -1
    :return: Array of weight values w
    """
    w = np.array([1., 1., 1.])  # Initialize weight array as 1's

    iter_count = 0  # Counter to track number of iterations passed
    update_count = 0  # Counter to track number of times a weight was updated
    misclassified = False  # Boolean check if there is misclassified data

    while True:  # Loop until no misclassified date is found
        for i, d in enumerate(data):  # i = iterations, d = data point
            if (np.dot(d, w) * output[i]) <= 0:  # Check for misclassification
                misclassified = True
                # Update weight values
                w[0] = w[0] + d[0] * output[i]
                w[1] = w[1] + d[1] * output[i]
                w[2] = w[2] + d[2] * output[i]
                update_count += 1
        iter_count += 1

        # Reset boolean check if misclassified data was found
        if misclassified is True:
            misclassified = False
        # Break loop if no misclassified data was found
        elif misclassified is False:
            break

    # Number of iterations needed
    print('Number of iterations needed:', iter_count)

    # Number of weight updates needed
    print('Number of weight updates needed:', update_count)

    return w


def main():
    """
    main function
    :return: none
    """
    # Generate the initial data set
    # data_points = np.random.rand(D, 3) * RANGE  # Array of size Dx3
    print("For %i points," % D)
    dataset = np.random.multivariate_normal(mean=[MEAN_X, MEAN_Y], cov=np.diag([3, 3]), size=D)
    dataset = np.insert(dataset, 0, 1, axis=1)  # Insert column for bias value

    y_output = np.array([])  # Generate empty array for output y
    for i in dataset:
        y_output = np.append(y_output, h(i, tf))  # Append y-output values

    # Run PLA on data to determine best-fit target function g, return weights
    # as w
    w = perceptron_learning_algorithm(dataset, y_output)

    # Print weight values
    # print("w0: ", w[0])
    # print("w1: ", w[1])
    # print("w2: ", w[2])

    line_parameters = [-w[1]/w[2], -w[0]/w[2]]

    print("Target function:\n\t\ty =", tf[0], "* x +", tf[1])
    print("Prediction func:\n\t\ty = %s" % float('%.1g' % line_parameters[0]),
          "* x + %s" % float('%.1g' % line_parameters[1]))

    plot(dataset, y_output, line_parameters, show_tf=True)


if __name__ == "__main__":
    main()
