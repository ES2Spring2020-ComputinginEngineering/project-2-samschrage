The nearest neighbor approach runs by using the normalized glucose, hemoglobin, and classification data that is provided.
Once that is inputed, the script will create a random point and classify it according to its nearest neighbor.
The same approach is used for the K nearest neighbor except it classifies according to the k nearest neighbors
where k must be provided in the function KNearestNeighborClassifier.


The k means clustering code normalizes the data, and then defines the centroids. It does this with the function
defineCentroid(k)
which takes a number k
returns k hemoglobin and k glucose points to use as centroids. 

It then sets the previous centroids to 0,0 in order to create a condition for the while loop to start. Inside the while loop, 
which runs while the previous centroid does not equal the current centroid, it calls the function 

assignLabels(hemoglobin_centroids, glucose_centroids, glucose, hemoglobin)
Classifies each point based on which centroids it is closest to
returns an array which classifies every given point with either a 0, 1, 2... dependeing on the k from the previous functions

It then graphs the points with their assigned classification.
The current centroids are set to the previous centroids, and 

updateCentroidLocation(glucose, hemoglobin, classifier, k)
updates the centroid location to the average position of the points classified by each centroid. 
returns k new glucose and hemoglobin values that are the new centroids

It then checks to see if the current centroids equal the previous centroids
and if they are equal the loop stops.
It then analyzes the results in order to determine the percentages of information that the k means clustering algorithim got right
based on the actual provided data.