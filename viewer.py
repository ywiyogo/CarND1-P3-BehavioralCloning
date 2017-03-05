""" Viewer function """

from helper_functions import plot_history
from os.path import isfile
import sys
import pickle

fname_hist = sys.argv[1]
if isfile(fname_hist):
    print("History data found")
    with open(fname_hist, 'rb') as f:
        h_data = pickle.load(f)
        # Show history
        plot_history(h_data)
else:
    print("File not found")
