from sklearn.neighbors import NearestNeighbors
import numpy as np
from itertools import cycle

class xyz(object):
    def __init__(self, model, poi, data):
        self.model = model
        self.poi = poi
        self.data = data
        
    def feasible_recourse_actions(self, K = 10):
        sorted_closest_points = np.array(sorted([(np.linalg.norm((self.data[i] - self.poi).toarray()), self.data[i]) for i in range(self.data.shape[0])], key = lambda row: row[0]))[: K, 1]
        K_nearest_points = list()
        k_counter = 0
        for point in sorted_closest_points:
            if self.model.predict(self.poi) != self.model.predict(point):
                K_nearest_points.append(point)
                k_counter += 1
                if k_counter == K:
                    break
        return np.array(K_nearest_points)

    def value(self, S, K):
        fra = self.feasible_recourse_actions(K)
        for points in fra:
            xp = list()
            for i in range(points.shape[0]):
                if i in S:
                    xp.append(points[i])
                else:
                    xp.append(self.poi[i])
            if self.model.predict(xp) != self.model.predict(self.poi):
                return 1
        return 0


def main():
   pass

if __name__ == "__main__":
    main()
    
    


        
