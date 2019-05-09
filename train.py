#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint, sample
import pickle
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from projectUtils import loadImagesFromDir, windowRecoveringArea, extractHOGFeatures, suppNonMaxima, loadModel, imagesDetection, imageDetection

def randomNegWindows(image, scale, H, L, posWindows, N):
    negSubImage = np.zeros((N, H, L), dtype=float)
    negWindows = np.zeros((N, 4), dtype=int)
    maxScale = 0
    for s in scale:
        if image.shape[0]-H*s > 0 and image.shape[1]-L*s > 0:
            maxScale += 1
    i = 0
    while i < N:
        r = scale[randint(0, maxScale-1)]
        realH = H*r
        realL = L*r
        realI = randint(0, int(image.shape[0]-realH-1))
        realJ = randint(0, int(image.shape[1]-realL-1))
        window = [realI, realJ, realH, realL]
        isValid = True
        for win in np.concatenate((posWindows,negWindows)):
            isValid = isValid and windowRecoveringArea(window, win) < 0.5
            if not isValid:
                break
        if isValid:
            imResized = resize(image, (round(image.shape[0]/r), round(image.shape[1]/r)), anti_aliasing=True, mode="reflect")
            windowI = int(realI/r)
            windowJ = int(realJ/r)
            subImage = imResized[windowI:(windowI+H), windowJ:(windowJ+L)]
            negSubImage[i] = subImage
            negWindows[i] = window
            i += 1
    return [negWindows, negSubImage]

def extractWindowImages(images, windows, H, L):
    windowsImages = np.zeros((windows.shape[0], H, L))
    i = 0
    for w in windows:
        [imIdx, x, y, h, l] = w
        imResized = resize(images[imIdx][x:(x+h), y:(y+l)], (H, L), anti_aliasing=True, mode="reflect")
        windowsImages[i] = imResized
        i += 1
    return windowsImages

def falsePositifWindows(guessedWins, realWins):
    falsePositif = np.zeros((1,6))
    for gWin in guessedWins:
        isFalsePos = True
        for rWin in realWins[realWins[:,0]==gWin[0]]:
            if windowRecoveringArea(rWin[1:5], gWin[1:5]) > 0.6:
                isFalsePos = False
                break
        if isFalsePos:
            falsePositif = np.concatenate((falsePositif, gWin.reshape((1,6))))
    if falsePositif.shape[0] == 1:
        return None
    else:
        return falsePositif[1:,:]

def getPosImages(trainImages, posWindows, H, L):
    trainPosImages = []
    i = 0
    for w in posWindows:
        [imIdx, x, y, h, l] = w
        im = resize(trainImages[imIdx-1][x:(x+h), y:(y+l)], (H, L), anti_aliasing=True, mode="reflect")
        trainPosImages += [im]
        #ajout de symetrique
        trainPosImages += [im[:,::-1]]

        #ajout de rotation petit angle
        theta = 15
        
        # Rotation de rectangle pythagore trigo tout ça ...
        # permet d'avoir une image sans zone noir sans pixel en dehors de l'image originale
        # probleme si padding = 0 donc on utilise math.ceil pour avoir au moins 1 de padding
        padding_left = math.ceil( (l/2)*math.cos(math.radians(theta)) + (h/2)*math.sin(math.radians(theta)))
        padding_left = math.ceil( padding_left - l/2)


        padding_top = math.ceil(( h/2)*math.cos(math.radians(theta)) + (l/2)* math.sin(math.radians(theta)))
        padding_top = math.ceil( padding_top - h/2)

        realI = max(0,x-padding_top)
        realH = min(trainImages[imIdx-1].shape[0],(x+h)+padding_top)

        realJ = max(0,y-padding_left)
        realL = min(trainImages[imIdx-1].shape[1],(y+l)+padding_left)

        im = rotate(trainImages[imIdx-1][realI:realH, realJ:realL],angle=theta)
        im = im[x - realI:(x+h) - realH, y- realJ:(y+l) -realL]
        
        im = resize(im,(H,L),anti_aliasing=True, mode="reflect")
        
        trainPosImages += [im]
        #ajout de symetrique
        trainPosImages += [im[:,::-1]]
        i+=1
    return np.array(trainPosImages)
    
def saveModel(clf, file):
    with open(file, 'wb') as file:  
        pickle.dump(clf, file)
    
if __name__ == "__main__":
    H = 90
    L = 60
    randomNegScale = np.arange(0.7, 4, 0.3)
    detectionScale = np.arange(0.5, 6, 0.3)
    negPerImage = 16
    
    modelFile = "model.pkl"                                                                                                                                                                                
    if len(sys.argv)>1:
        modelFile = sys.argv[1]
        
    featureExtractor = extractHOGFeatures

    print("Loading images")

    images = loadImagesFromDir("./data/train/")
    posWindows = np.loadtxt("./data/label.txt", dtype=int)

    print("Extracting positive features")

    posImages =  getPosImages(images, posWindows, H, L)
    posFeatures = np.zeros((posImages.shape[0], 3645), dtype=float)
    for i in range(posImages.shape[0]):
        posFeatures[i] = featureExtractor(posImages[i])
    
    print("Using %i positive features"%posFeatures.shape[0])
    
    posWindows[:,0] = posWindows[:,0] - 1 
    print("Extracting random negative features")

    negFeatures = np.zeros((images.shape[0]*negPerImage, 3645), dtype=float)
    negFeaturesIdx = 0
    for imIdx in range(images.shape[0]):
        [negWin, negSubImage] = randomNegWindows(images[imIdx], randomNegScale, H, L, posWindows[posWindows[:,0]==imIdx,1:], negPerImage)
        for im in negSubImage:
            negFeatures[negFeaturesIdx] = featureExtractor(im)
            negFeaturesIdx += 1
    
    print("Using %i negative features"%negFeatures.shape[0])

    features = np.concatenate((posFeatures, negFeatures))
    posLabels = np.ones((posFeatures.shape[0]))
    negLabels = -np.ones((negFeatures.shape[0]))
    labels = np.concatenate((posLabels, negLabels))

    print("1st model training with cross-validation")

    paramGrid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }
    clf = SVC(kernel="rbf")
    clf = GridSearchCV(clf, paramGrid, scoring="balanced_accuracy", cv=5)
    clf.fit(features,labels)
    
    print("Evaluating 1st model on train images")
    detections = imagesDetection(images, H, L, detectionScale, clf, featureExtractor)
    
    print("Extracting false positive features")
    falsePosWin = falsePositifWindows(detections, posWindows)
    w = np.array(falsePosWin[:,:-1], dtype=int)
    falsePosImages = extractWindowImages(images, w, H, L)
    for fPos in falsePosImages:
        fPosFeatures = np.array(featureExtractor(fPos)).reshape((1,3645))
        features = np.concatenate((features, fPosFeatures))
        labels = np.concatenate((labels, np.array([-1])))
    
    print("Re-training model with %i false positif features added"%falsePosImages.shape[0])
    clf2 = SVC(kernel="rbf", gamma=clf.best_prams_["gamma"], C=clf.best_prams_["C"])
    clf2.fit(features, labels)
    print("Saving model to file %s"%modelFile)
    saveModel(clf2, modelFile)
