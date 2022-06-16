import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

X = np.load('image.npz')["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nClasses = len(classes)

sample_per_class = 5
figure = plt.figure(figsize=(nClasses * 2, (1 + sample_per_class * 2)))

idx_cls = 0
for cls in classes:
    idxs = np.flatnonzero(y == cls)
    idxs = np.random.choice(idxs, sample_per_class, replace=False)
    i = 0

    for idx in idxs:
        plt_idx = i * nClasses + idx_cls + 1
        p = plt.subplot(sample_per_class, nClasses, plt_idx)
        p = sns.heatmap(np.reshape(X[idx], (22, 30)), cmap=plt.cm.gray,
                        xticklabels=False, yticklabels=False, cbar=False)
        p = plt.axis("off")
        i += 1
    idx_cls += 1

print("IDX Completed")

xTrain, xTest, yTrain, yTest = tts(
    X, y, random_state=9, train_size=7500, test_size=2500)
xTrainScaled = xTrain/255.0
xTestScaled = xTest/255.00

print("x and y train completed.")

cls = LogisticRegression(
    solver="saga", multi_class="multinomial").fit(xTrainScaled, yTrain)
print("Cls done !!!")

yPred = cls.predict(xTestScaled)
accuracy = accuracy_score(yTest, yPred)
print("Accuracy:", accuracy)

cm = pd.crosstab(yTest, yPred, rownames=["Actual"], colnames=["Predicted"])
p = plt.figure(figsize=(10, 10))
p = sns.heatmap(cm, annot=True, cbar=False, fmt="d")
print("Heatmap visible")

cap = cv2.VideoCapture(0)

while(True):
    try:
        rect, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upperLeft = (int(width/2 - 56), int(height/2-56))
        bottomRight = (int(width/2+56), int(height/2+56))

        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)

        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]

        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert("L")
        image_bw_resize = image_bw.resize((28, 28), Image.Resampling.LANCZOS)
        image_bw_resize_inverted = PIL.ImapeOps.invert(image_bw_resize)

        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resize_inverted, pixel_filter)
        image_bw_resize_inverted_scaled = np.clip(
            image_bw_resize_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resize_inverted)

        image_bw_resize_inverted_scaled = np.asarray(
            image_bw_resize_inverted_scaled) / max_pixel

        test_sample = np.array(image_bw_resize_inverted_scaled).reshape(1, 784)

        test_pred = cls.predict(test_sample)
        print("Predicted Classes: ", test_pred)
        cv2.imshow("frame", gray)

        if(cv2.waitKey(1) and 0xFF == ord("q")):
            break

    except Exception as e:
        pass
