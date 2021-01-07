import pandas as ass
import numpy as np

data = ass.read_excel("Course Evaluation .xlsx")

T = data.iloc[:, 1:].values


colors = 10 * ["g", "r", "c", "b", "k"]


class mean:
    def __init__(self, k=2, total=0.001, maximum_iteration=300):
        self.k = k
        self.total = total
        self.maximum_iteration = maximum_iteration

    def operation(self, data):

        self.centroids = {}

        for j in range(self.k):
            self.centroids[j] = data[j]

        for j in range(self.maximum_iteration):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for newset in T:
                distance = [np.linalg.norm(newset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distance.index(min(distance))
                self.classifications[classification].append(newset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.total:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distance = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distance.index(min(distance))
        return classification


clf = mean()
clf.operation(T)
print(clf.centroids)
print(clf.classifications)








