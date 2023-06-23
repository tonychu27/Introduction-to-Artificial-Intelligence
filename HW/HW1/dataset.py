import os
import cv2
import numpy as np


def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []

    pathofFace = dataPath+'/face/'
    pathofNonface = dataPath+'/non-face/'

    for image in os.listdir(pathofFace):
        if image.endswith('.pgm'):
            Path = os.path.join(pathofFace, image)
            img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
            dataset.append((img, 1))

    for image in os.listdir(pathofNonface):
        if image.endswith('.pgm'):
            Path = os.path.join(pathofNonface, image)
            img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
            dataset.append((img, 0))

    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
