"""
Author: YWiyogo
Description: Helper function for P3
"""
import matplotlib.pyplot as plt
import numpy as np
# Analyzing the training datasets
def show_histogram(data, title="Histogram of the datasets"):
    """
    Plotting histogram
    """
    fig_hist = plt.figure(figsize=(15, 8))
    ax = fig_hist.add_subplot(111)
    ax.hist(data, rwidth=0.8, align="mid", zorder=3)
    ax.yaxis.grid(True, linestyle='--', zorder=0)
    ax.set_ylabel('Occurrences')
    ax.set_xlabel('Steering angle')
    ax.set_title(title)
    plt.show()

def get_relative_path(abs_path):
    """
    Get the relative path of the image training data
    """
    filename = abs_path.split("/")[-1]
    return "../p3_training_data/IMG/" + filename

def plot_history(history):
    """
    Plot function for model history object which contains the loss values
    """
    plt.figure()
    plt.plot(np.arange(1, len(history['loss']) + 1), np.array(history['loss']))
    plt.plot(np.arange(1, len(history['val_loss']) + 1), np.array(history['val_loss']))
    plt.title("Model History")
    plt.ylabel("MSE loss")
    plt.legend(['training', 'validation'], loc='upper right')
    plt.xlabel("Epoch")
    plt.grid(True, linestyle='--')
    plt.show()
