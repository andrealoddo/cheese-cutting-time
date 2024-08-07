import matplotlib.pyplot as plt


def create_scatter_plot(predicted_values, actual_values, print_plot=False):
    fig = plt.figure()
    plt.plot([predicted_values.min(), predicted_values.max()], [actual_values.min(), actual_values.max()], 'r--')
    plt.scatter(actual_values, predicted_values, color='b')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    if print_plot:
        plt.show()

    return fig


def create_residual_plot(predicted_values, actual_values):
    fig = plt.figure()
    residuals = actual_values - predicted_values
    plt.scatter(predicted_values, residuals, color='b', label='Residuals')
    plt.plot([predicted_values.min(), predicted_values.max()], [0, 0], 'r--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

    return fig
