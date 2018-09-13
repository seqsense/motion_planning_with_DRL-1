import csv
import numpy as np

def raycasting_from_dataset(self,pose):
    with open('raycasting_test.csv','r') as f:
        reader = csv.reader(f)
        x = 0.0123 - 0.0123%0.01
        y = 0.0345 - 0.0345%0.01
        theta = 0.00
        lidar = np.zeros(36) 
        for row in reader:
            if float(row[0]) == x and float(row[1]) == y and float(row[2]) == theta:
                for i in range(36):
                    lidar[i] = row[i+3]
                    return lidar
