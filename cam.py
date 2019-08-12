import numpy as np
import cv2
from sklearn import tree
from sklearn.model_selection import train_test_split


def ReadData(dataFile):
    # Data in format [B G R Label] from
    data = np.genfromtxt(dataFile, dtype=np.int32)

    labels = data[:, 3]
    data = data[:, 0:3]

    return data, labels


def BGR2HSV(bgr):
    bgr = np.reshape(bgr, (bgr.shape[0], 1, 3))
    hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv = np.reshape(hsv, (hsv.shape[0], 3))

    return hsv


def trainTree(data, labels, useHSV):
    if (useHSV):
        data = BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)

    # print(trainData.shape)
    # print(trainLabels.shape)
    # print(testData.shape)
    # print(testLabels.shape)
    #
    # print(clf.feature_importances_)
    # print(clf.score(testData, testLabels))

    return clf


def applyToImage_old(image, flUseHSVColorspace):
    data, labels = ReadData()
    clf = TrainTree(data, labels, flUseHSVColorspace)

    img = cv2.imread(path)
    print(img.shape)
    data = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    print(data.shape)

    if (flUseHSVColorspace):
        data = BGR2HSV(data)

    predictedLabels = clf.predict(data)

    imgLabels = np.reshape(predictedLabels, (img.shape[0], img.shape[1], 1))

    if (flUseHSVColorspace):
        cv2.imwrite('../results/result_HSV.png', ((-(imgLabels - 1) + 1) * 255))  # from [1 2] to [0 255]
    else:
        cv2.imwrite('../results/result_RGB.png', ((-(imgLabels - 1) + 1) * 255))


def applyToImage(img, classifier):
    data = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    predictedLabels = classifier.predict(data)
    imgLabels = np.reshape(predictedLabels, (img.shape[0], img.shape[1], 1))
    return ((-(imgLabels - 1) + 1) * 255).astype(np.uint8)  # from [1 2] to [0 255]


def main():
    data, labels = ReadData('data/Skin_NonSkin.txt')
    hsvClassifier = trainTree(data, labels, useHSV=True)
    bgrClassifier = trainTree(data, labels, useHSV=False)

    cap = cv2.VideoCapture(0)
    while 1:
        ret, bgr = cap.read()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsvLabelsImg = applyToImage(hsv, hsvClassifier)
        bgrLabelsImg = applyToImage(bgr, bgrClassifier)

        cv2.imshow('frame', bgr)
        cv2.imshow('bgrLabelsImg', bgrLabelsImg)
        cv2.imshow('hsvLabelsImg', hsvLabelsImg)
        if cv2.waitKey(30) == 27:
            break
    cap.release()


def main():
    data, labels = ReadData('data/Skin_NonSkin.txt')
    hsvClassifier = trainTree(data, labels, useHSV=True)
    bgrClassifier = trainTree(data, labels, useHSV=False)

    bgr = cv2.imread('face.png')
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsvLabelsImg = applyToImage(hsv, hsvClassifier)
    bgrLabelsImg = applyToImage(bgr, bgrClassifier)

    cv2.imshow('frame', bgr)
    cv2.imshow('bgrLabelsImg', bgrLabelsImg)
    cv2.imshow('hsvLabelsImg', hsvLabelsImg)
    cv2.waitKey()


main()
