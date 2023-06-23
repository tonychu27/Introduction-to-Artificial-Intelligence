import os
import cv2
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    Data = list(open(dataPath, "r"))
    line = 0
    while line < len(Data):
        Name, num = map(str, Data[line].split())
        line += 1
        people = []
        for k in range(int(num)):
            people.append(tuple(map(int, Data[line].split())))
            line += 1
        img = cv2.imread(os.path.join("data/detect/", Name))
        imgGray = cv2.imread(
            os.path.join("data/detect/", Name), cv2.IMREAD_GRAYSCALE)
        for person in people:
            x, y, w, h = person
            faceRegion = imgGray[y:y+h, x:x+w]
            resizedFace = cv2.resize(
                faceRegion, (19, 19), interpolation=cv2.INTER_LINEAR)
            prediction = clf.classify(resizedFace)

            if prediction == 1:
                cv2.rectangle(img, (x, y), (x+w, y+h),
                              color=(0, 255, 0), thickness=5)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h),
                              color=(0, 0, 255), thickness=5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(img)
        plt.show()
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
