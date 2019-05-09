#!/usr/bin/python3

import numpy as np
import sys
from projectUtils import loadImagesFromDir, imagesDetection, extractHOGFeatures, loadModel

if __name__ == "__main__":
    H = 90
    L = 60
    scale = np.arange(0.7, 4, 0.3)
    
    featureExtractor = extractHOGFeatures
    
    detectionsFile = "detections.txt"
    modelFile = "model.pkl"
    if len(sys.argv)>1:
        modelFile = sys.argv[1]
    if len(sys.argv)>2:    
        detectionsFile = sys.argv[2]
        
    clf = loadModel(modelFile)
    
    print("Analyzing images")
    images = loadImagesFromDir("./data/test/")
    detections = imagesDetection(images, H, L, scale, clf, featureExtractor)
    
    detections[:,0] = detections[:,0] + 1
    
    print("Writing results to %s"%detectionsFile)
    with open(detectionsFile, 'w') as file:
        for d in detections:
            s = "%i %i %i %i %i %f\n"%(int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), round(d[5], 2))
            file.write(s)