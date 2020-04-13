#Sam Schrage
#Project 2
# I worked alone on this project
#k means clustering main script
import KMeansClustering_functions as kmc #Use kmc to call your functions
import numpy as np

glucose, hemoglobin, classification=kmc.openckdfile()
k=2
glucose, hemoglobin=kmc.normalizeData(glucose, hemoglobin, classification)
hemoglobin_centroids, glucose_centroids=kmc.defineCentroid(k)
previous_hemoglobin_centroids=np.zeros(k)
previous_glucose_centroids=np.zeros(k)
test=False
test1=False
count=0
while test==False and test1==False:
    classifier=kmc.assignLabels(hemoglobin_centroids, glucose_centroids, glucose, hemoglobin) 
    kmc.graphingKMeans(glucose, hemoglobin, classifier, hemoglobin_centroids, glucose_centroids)
    previous_hemoglobin_centroids=hemoglobin_centroids
    previous_glucose_centroids=glucose_centroids
    glucose_centroids, hemoglobin_centroids=kmc.updateCentroidLocation(glucose, hemoglobin, classifier, k) 
    test=np.array_equal(previous_hemoglobin_centroids, hemoglobin_centroids)
    test1=np.array_equal(previous_glucose_centroids, glucose_centroids)
    count=count+1
    if count==50:
        break
print("Glucose Centroids:",glucose_centroids)
print("Hemoglobin Centroids:",hemoglobin_centroids)
CKD_actual=np.count_nonzero(classification)
non_CKD_actual=158-CKD_actual
True_positive_rate=classifier[:CKD_actual]
True_positive_divisor=len(True_positive_rate)
True_positive_ones=np.count_nonzero(True_positive_rate)
True_positive_zeros=True_positive_divisor-True_positive_ones
if True_positive_ones>True_positive_zeros:
    True_positive_rate=True_positive_ones/True_positive_divisor
else:
    True_positive_rate=True_positive_zeros/True_positive_divisor
print("True Positive Rate:",True_positive_rate)
False_negative_rate=1-True_positive_rate
print("False Negative Rate:",False_negative_rate)
True_negative_rate=classifier[CKD_actual:]
True_negative_divisor=len(True_negative_rate)
True_negative_ones=np.count_nonzero(True_negative_rate)
True_negative_zeros=True_negative_divisor-True_negative_ones
if True_negative_ones>True_negative_zeros:
    True_negative_rate=True_negative_ones/True_negative_divisor
else:
    True_negative_rate=True_negative_zeros/True_negative_divisor
print("True Negative Rate:",True_negative_rate)
False_positive_rate=1-True_negative_rate
print("False Positive Rate:",False_positive_rate)