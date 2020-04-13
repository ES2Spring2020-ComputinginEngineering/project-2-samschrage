#Sam Schrage
#Project 2
# I worked alone on this project
#Nearest Neighbor and K Nearest Neighbor

import numpy as np
import matplotlib.pyplot as plt
import random
import math


# FUNCTIONS
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def normalizeData(glucose, hemoglobin, classification):
    glucose_scaled=(glucose-70)/(490-70)
    hemoglobin_scaled=(hemoglobin-3.1)/(17.8-3.1)
    return glucose_scaled, hemoglobin_scaled

def graphData(glucose_scaled, hemoglobin_scaled, classification, title):
    plt.figure()
    plt.plot(hemoglobin_scaled[classification==1],glucose_scaled[classification==1], "k.", label = "Class 1")
    plt.plot(hemoglobin_scaled[classification==0],glucose_scaled[classification==0], "r.", label = "Class 0")
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.title(str(title))
    plt.show()
    
def createTestCase():
    newhemoglobin=np.random.rand(1,1)
    newglucose=np.random.rand(1,1)
    return newhemoglobin, newglucose

def calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin):
    index=len(glucose)
    distance=np.zeros((index))
    for i in range(index):
        distance[i]=math.sqrt((newhemoglobin-hemoglobin[i])**(2)+(newglucose-glucose[i])**(2))
    return distance

def nearestNeighborClassifier(newglucose, newhemoglobin, glucose, hemoglobin, classification):
    distance=calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)
    min_index=np.argmin(distance)
    classification_new_point=classification[min_index]
    return classification_new_point

def graphTestCase(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification, color, classifier, title):
    plt.figure()
    plt.plot(hemoglobin_scaled[classification==1],glucose_scaled[classification==1], "k.", label = "Class 1")
    plt.plot(hemoglobin_scaled[classification==0],glucose_scaled[classification==0], "r.", label = "Class 0")
    plt.plot(newhemoglobin,newglucose, color, label = classifier, markersize=20)
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.title(str(title))
    plt.show()
    
def kNearestNeighborClassifier(k, newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification):
    distance_k=calculateDistanceArray(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled)
    sorted_indices=np.argsort(distance_k)
    k_indices=sorted_indices[:k]
    k_classifications=classification[k_indices]
    classification_new_point_k=np.median(k_classifications)
    return classification_new_point_k
    


# MAIN SCRIPT
glucose, hemoglobin, classification = openckdfile()
glucose_scaled, hemoglobin_scaled=normalizeData(glucose, hemoglobin, classification)
title="Original Data"
graphData(glucose_scaled, hemoglobin_scaled, classification, title)
newhemoglobin, newglucose=createTestCase()
classification_new_point=nearestNeighborClassifier(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification)
classifier="New Data"
if classification_new_point==1:
    color="k."
else:
    color="r."
title="Nearest Neighbor"
graphTestCase(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification, color, classifier, title)

classification_new_point_k=kNearestNeighborClassifier(10, newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification)
if classification_new_point_k==1:
    color1="k."
else:
    color1="r."
title="K Nearest neighbor"
graphTestCase(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification, color1, classifier, title)
