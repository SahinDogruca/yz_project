import numpy as np
from scipy.spatial import distance

def generate_A(num=1000):
  X = []
  y = []
  for _ in range(num):
    matrix = np.zeros((25,25), dtype='i1')
    points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(2)]
    while len(set(points)) != len(points):
      points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(2)]
    for i in range(len(points)):
      matrix[points[i][0]][points[i][1]] = 1

    dist = distance.euclidean(points[0], points[1])
    X.append(matrix)
    y.append(dist)

  return np.array(X), np.array(y)

def generate_B(num=1000):
  X = []
  y = []
  for _ in range(num):
    matrix = np.zeros((25,25), dtype='i1')
    point_count = np.random.randint(3,10)
    points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(point_count)]
    while len(set(points)) != len(points):
      points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(point_count)]
    for i in range(len(points)):
      matrix[points[i][0]][points[i][1]] = 1

    min_euclidian = 1000
    points.sort(key=lambda x: x[0])
    for i in range(len(points)):
      for j in range(i+1, len(points)):
        euclidian = distance.euclidean(points[i], points[j])
        min_euclidian = min(euclidian, min_euclidian)

    X.append(matrix)
    y.append(min_euclidian)

  return np.array(X), np.array(y)


def generate_C(num=1000):
  X = []
  y = []
  for _ in range(num):
    matrix = np.zeros((25,25), dtype='i1')
    point_count = np.random.randint(3,10)
    points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(point_count)]
    while len(set(points)) != len(points):
      points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(point_count)]
    for i in range(len(points)):
      matrix[points[i][0]][points[i][1]] = 1

    max_euclidian = 0
    points.sort(key=lambda x: x[0])
    for i in range(len(points)):
      for j in range(i+1, len(points)):
        euclidian = distance.euclidean(points[i], points[j])
        max_euclidian = max(euclidian, max_euclidian)

    X.append(matrix)
    y.append(max_euclidian)

  return np.array(X), np.array(y)


def generate_D(num=1000):
  X = []
  y = []
  for _ in range(num):
    matrix = np.zeros((25,25), dtype='i1')
    point_count = np.random.randint(1,10)
    points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(point_count)]
    while len(set(points)) != len(points):
      points = [(np.random.randint(0,24), np.random.randint(0,24)) for _ in range(point_count)]
    for i in range(len(points)):
      matrix[points[i][0]][points[i][1]] = 1

    X.append(matrix)
    y.append(point_count)

  return np.array(X), np.array(y)



def generate_E(num=1000):
  X = []
  y = []
  for _ in range(num):
    matrix = np.zeros((25,25), dtype="i1")
    sqr_count = np.random.randint(1,10)

    for _ in range(sqr_count):

      sqr_dim = np.random.randint(2,10)

      max_x= max_y = 25-sqr_dim
      rand_x, rand_y = np.random.randint(0,max_x), np.random.randint(0,max_y)
      matrix[rand_x:rand_x+sqr_dim,rand_y:rand_y+sqr_dim] = 1

    X.append(matrix)
    y.append(sqr_count)

  return np.array(X), np.array(y)

