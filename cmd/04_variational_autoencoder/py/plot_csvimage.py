import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def predict_plot(data_file):
    print(f"Attempting to load: {data_file}")
    image_data = np.loadtxt(data_file, delimiter=',', dtype=np.float64)
    plt.imshow(image_data, cmap='gray')
    plt.show()

if __name__ == "__main__":
    predict_plot("test.csv")