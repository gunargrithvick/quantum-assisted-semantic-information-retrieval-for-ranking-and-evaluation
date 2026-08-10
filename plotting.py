"""Headless-safe plot display helpers."""
import matplotlib.pyplot as plt

def display_current_plot():

    backend=plt.get_backend().lower()

    if "agg" not in backend:
        plt.show()

    plt.close()
