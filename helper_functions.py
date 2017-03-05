import matplotlib.pyplot as plt
import numpy as np
# Analyzing the training datasets
def show_histogram(data, title="Histogram of the datasets"):
    """
    Plotting histogram
    """
    fig_hist = plt.figure(figsize=(15, 8))
    ax = fig_hist.add_subplot(111)
    ax.hist(data, bins=27, rwidth=0.8, align="mid", zorder=3)
    ax.yaxis.grid(True, linestyle='--', zorder=0)
    ax.set_xticks(np.arange(-1.3, 1.3, 0.1))
    ax.set_ylabel('Occurrences')
    ax.set_xlabel('Steering angle')
    ax.set_title(title)
    plt.show()

def get_relative_path(abs_path):
    filename = abs_path.split("/")[-1]
    return "../p3_training_data/IMG/" + filename

def plot_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model History')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
