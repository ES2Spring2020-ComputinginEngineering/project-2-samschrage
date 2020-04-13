#Sam Schrage
#Project 2
# I worked alone on this project
#K means clustering functions

import numpy as np
import matplotlib.pyplot as plt
import random
import math

def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def normalizeData(glucose, hemoglobin, classification):
    glucose_scaled=(glucose-70)/(490-70)
    hemoglobin_scaled=(hemoglobin-3.1)/(17.8-3.1)
    return glucose_scaled, hemoglobin_scaled

def defineCentroid(clusters):
    hemoglobin_centroids=np.random.rand(clusters, 1)
    glucose_centroids=np.random.rand(clusters, 1)
    return hemoglobin_centroids, glucose_centroids

def calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin):
    index=len(glucose)
    distance=np.zeros((index))
    for i in range(index):
        distance[i]=math.sqrt((newhemoglobin-hemoglobin[i])**(2)+(newglucose-glucose[i])**(2))
    return distance

def assignLabels(hemoglobin_centroids, glucose_centroids, glucose, hemoglobin):
    index1=len(hemoglobin_centroids)
    index2=len(glucose)
    distance=np.zeros((index2,index1))
    classifier=np.zeros((index2,1))
    for i in range(index1):
        glucose_centroid=glucose_centroids[i]
        hemoglobin_centroid=hemoglobin_centroids[i]
        distance[:,i]=calculateDistanceArray(glucose_centroid, hemoglobin_centroid, glucose, hemoglobin)
    minDistance=np.argmin(distance, axis=1)
    classifier=minDistance
    return classifier

def updateCentroidLocation(glucose, hemoglobin, classifier, k):
    index=len(glucose)
    new_centroid_glucose=np.zeros((index,k))
    new_centroid_hemoglobin=np.zeros((index,k))
    glucose_row=np.zeros(k)
    hemoglobin_row=np.zeros(k)
    for i in range(index):
        for j in range(k):
            if classifier[i]==j:
                new_centroid_glucose[i,j]=glucose[i]
                if glucose[i]==0:
                    new_centroid_glucose[i,j]=6
                    for s in range(k):
                        if s==j:
                            glucose_row[s]=glucose_row[s]+6
                new_centroid_hemoglobin[i,j]=hemoglobin[i]
                if hemoglobin[i]==0:
                    new_centroid_hemoglobin[i,j]=6
                    for t in range(k):
                        if t==j:
                            hemoglobin_row[t]=hemoglobin_row[t]+6
    hemoglobin_divisor=np.count_nonzero(new_centroid_hemoglobin, axis=0)
    glucose_divisor=np.count_nonzero(new_centroid_glucose, axis=0)
    new_centroid_glucose=np.sum(new_centroid_glucose, axis=0)
    new_centroid_hemoglobin=np.sum(new_centroid_hemoglobin, axis=0)
    new_centroid_glucose=new_centroid_glucose-glucose_row
    new_centroid_glucose=np.divide(new_centroid_glucose, glucose_divisor)
    new_centroid_hemoglobin=np.divide(new_centroid_hemoglobin, hemoglobin_divisor)
    return new_centroid_glucose, new_centroid_hemoglobin

def graphingKMeans(glucose, hemoglobin, assignment, hemoglobin_centroids, glucose_centroids):
    plt.figure()
    for i in range(assignment.max()+1):
        rcolor = np.random.rand(3,)
        plt.plot(hemoglobin[assignment==i],glucose[assignment==i], ".", label = "Class " + str(i), color = rcolor)
        plt.plot(hemoglobin_centroids[i], glucose_centroids[i], "D", label = "Centroid " + str(i), color = rcolor)
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.show()
