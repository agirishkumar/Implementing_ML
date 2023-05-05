import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UnivariateLinearRegression import UnivariateLinearRegression


def load_data(train_file, test_file):
    """Load training and testing data from CSV files."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df


def mean_squared_error(test_y, predictions):
    """Compute the mean squared error between true and predicted values."""
    return np.mean((test_y - predictions) ** 2)


def plot_data_and_predictions(train_df, reg, test_df, mse):
    """Plot training data, predictions as a line, and display the mean squared error."""
    plt.scatter(train_df['x'], train_df['y'], s=2, label='Training data')
    x_line = np.linspace(np.min(train_df['x']), 100, 500)
    y_line = reg.predict(x_line)
    plt.plot(x_line, y_line, color='red', label='Hypothesis line')
    plt.xlabel("Features (x)")
    plt.ylabel("Target (y)")
    plt.legend()
    plt.title(f"Mean Squared Error: {mse:.3f}")
    plt.show()


def main():
    train_df, test_df = load_data('linearregressiondata/train.csv', 'linearregressiondata/test.csv')
    reg = UnivariateLinearRegression(0.00005)
    reg.train(train_df['x'], train_df['y'])
    predictions = reg.predict(test_df['x'])

    mse = mean_squared_error(test_df['y'], predictions)
    print('Mean squared error:', mse)

    plot_data_and_predictions(train_df, reg, test_df, mse)


if __name__ == "__main__":
    main()
