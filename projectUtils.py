import numpy as np
import os
import pickle
from skimage import io, util
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from skimage.transform import pyramid_gaussian

def loadImagesFromDir(path):
    images = []
    if path[-1] != "/":
        path += "/"
    for f in sorted(os.listdir(path)):
        images.append(rgb2gray(util.img_as_float(io.imread(path+f))))
    return np.array(images)
    
def windowRecoveringArea(wi, wj):
    w = np.array([wi] + [wj])
    minX = np.argmin(w[:,1])
    minY = np.argmin(w[:,0])
    maxX = np.max(w[:,1])
    maxY = np.max(w[:,0])
    if w[minX,1] + w[minX,3] < maxX or w[minY,0] + w[minY,2] < maxY:
        return 0
    else:
        x = [maxX, maxY]
        y = [min([w[0,1] + w[0, 3],w[1,1] + w[1, 3]]), min([w[0,0] + w[0,2], w[1,0] + w[1,2]])]
        iArea = (y[0] - x[0])*(y[1] - x[1])
        return iArea / (w[0,2]*w[0,3] + w[1,2]*w[1,3] - iArea)
        
def extractHOGFeatures(im):
	return hog(im, feature_vector=True, block_norm="L2-Hys")
 
def plotWindows(image, windows, colors=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    i = 0
    for w in windows:
        if colors is None:
            rect = patch.Rectangle((w[1],w[0]),w[3],w[2],linewidth=1, edgecolor="r", facecolor='none')
        else:
            rect = patch.Rectangle((w[1],w[0]),w[3],w[2],linewidth=1, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
    plt.show()
        
def imagesDetection(images, H, L, scale, clf, featureExtractor, scoreDecision=0):
    detections = np.zeros((1, 5))
    imagesIdx = [] 
    for imIdx in range(images.shape[0]):
        if imIdx%100==0:
            print("analyzing image: %i"%imIdx)
        imDetections = imageDetection(images[imIdx], H, L, scale, clf, featureExtractor, scoreDecision)
        if imDetections is not None:
            nonMaxima = suppNonMaxima(imDetections)
            detections = np.concatenate((detections, nonMaxima))
            imagesIdx += [imIdx]*nonMaxima.shape[0]
    detections = detections[1:,:]
    imagesIdx = np.array(imagesIdx).reshape((detections.shape[0], 1))
    detections = np.concatenate((imagesIdx, detections), axis=1)    
    return detections

def imageDetection(im, H, L, scale, clf, featureExtractor, scoreDecision=0):
    maxScale = 0
    for s in scale:
        if im.shape[0]-H*s > 0 and im.shape[1]-L*s > 0:
            maxScale += 1
    windows = np.zeros((1,5), dtype=float)
    for s in scale[:maxScale-1]:
        imResized = resize(im, (round(im.shape[0]/s), round(im.shape[1]/s)), anti_aliasing=True, mode="reflect")
        realH = int(H*s)
        realL = int(L*s)
        i = 0
        j = 0
        while i < imResized.shape[0]-H:
            while j < imResized.shape[1]-L:
                features = [featureExtractor(imResized[i:(i+H), j:(j+L)])]
                score = clf.decision_function(features)
                if score > scoreDecision:
                    realI = int(i*s)
                    realJ = int(j*s)
                    w = np.array([realI, realJ, realH, realL,score], dtype=float)
                    windows = np.concatenate((windows, w.reshape((1,5))))
                j += 10
            i += 10
    if windows.shape[0]==1:
        return None
    else:
        return windows[1:,:]

def suppNonMaxima(detections):
    n = detections.shape[0]
    detections = detections[np.flip(detections[:,4].argsort(), axis=0),:]
    suppIdx = []
    for i in range(n):
        for j in range(i+1,n):
            if windowRecoveringArea(detections[i,:4], detections[j,:4]) > 0.25:
                suppIdx += [j]
    maximaIdx = [(i not in suppIdx) for i in range(n)]
    return detections[maximaIdx]
    
def loadModel(file):
    with open(file, 'rb') as file:  
        clf = pickle.load(file)
    return clf
