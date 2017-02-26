import matplotlib.pyplot as plt

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
