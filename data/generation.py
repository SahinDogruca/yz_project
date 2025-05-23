import numpy as np
from scipy.spatial import distance


def generate_A(num=1000):
    X = []
    y = []
    for _ in range(num):
        matrix = np.zeros((25, 25), dtype="i1")
        points = [
            (np.random.randint(0, 24), np.random.randint(0, 24)) for _ in range(2)
        ]
        while len(set(points)) != len(points):
            points = [
                (np.random.randint(0, 24), np.random.randint(0, 24)) for _ in range(2)
            ]
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
        matrix = np.zeros((25, 25), dtype="i1")
        point_count = np.random.randint(3, 10)
        points = [
            (np.random.randint(0, 24), np.random.randint(0, 24))
            for _ in range(point_count)
        ]
        while len(set(points)) != len(points):
            points = [
                (np.random.randint(0, 24), np.random.randint(0, 24))
                for _ in range(point_count)
            ]
        for i in range(len(points)):
            matrix[points[i][0]][points[i][1]] = 1

        min_euclidian = 1000
        points.sort(key=lambda x: x[0])
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                euclidian = distance.euclidean(points[i], points[j])
                min_euclidian = min(euclidian, min_euclidian)

        X.append(matrix)
        y.append(min_euclidian)

    return np.array(X), np.array(y)


def generate_C(num=1000):
    X = []
    y = []
    for _ in range(num):
        matrix = np.zeros((25, 25), dtype="i1")
        point_count = np.random.randint(3, 10)
        points = [
            (np.random.randint(0, 24), np.random.randint(0, 24))
            for _ in range(point_count)
        ]
        while len(set(points)) != len(points):
            points = [
                (np.random.randint(0, 24), np.random.randint(0, 24))
                for _ in range(point_count)
            ]
        for i in range(len(points)):
            matrix[points[i][0]][points[i][1]] = 1

        max_euclidian = 0
        points.sort(key=lambda x: x[0])
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                euclidian = distance.euclidean(points[i], points[j])
                max_euclidian = max(euclidian, max_euclidian)

        X.append(matrix)
        y.append(max_euclidian)

    return np.array(X), np.array(y)


def generate_D(num=1000):
    X = []
    y = []
    for _ in range(num):
        matrix = np.zeros((25, 25), dtype="i1")
        point_count = np.random.randint(1, 10)
        points = [
            (np.random.randint(0, 24), np.random.randint(0, 24))
            for _ in range(point_count)
        ]
        while len(set(points)) != len(points):
            points = [
                (np.random.randint(0, 24), np.random.randint(0, 24))
                for _ in range(point_count)
            ]
        for i in range(len(points)):
            matrix[points[i][0]][points[i][1]] = 1

        X.append(matrix)
        y.append(point_count)

    return np.array(X), np.array(y)


def generate_E(num=1000):
    X = []
    Y = []
    for _ in range(num):
        matrix = np.zeros((25, 25), dtype="i1")
        sqr_count = np.random.randint(1, 10)
        squares = []  # List to track (x, y, dimension) of each square

        for _ in range(sqr_count):
            placed = False
            attempts = 0
            while not placed and attempts < 100:  # Prevent infinite loops
                sqr_dim = np.random.randint(2, 10)
                max_x = 25 - sqr_dim
                max_y = 25 - sqr_dim
                if max_x <= 0 or max_y <= 0:
                    continue  # Skip if dimension is too large
                rand_x = np.random.randint(0, max_x)
                rand_y = np.random.randint(0, max_y)

                # Check for overlap or subset with existing squares
                valid = True
                for x, y, d in squares:
                    # Check for overlap
                    overlap = not (
                        rand_x + sqr_dim <= x
                        or x + d <= rand_x
                        or rand_y + sqr_dim <= y
                        or y + d <= rand_y
                    )
                    # Check if new is subset of existing
                    new_subset = (
                        rand_x >= x
                        and rand_x + sqr_dim <= x + d
                        and rand_y >= y
                        and rand_y + sqr_dim <= y + d
                    )
                    # Check if existing is subset of new
                    existing_subset = (
                        x >= rand_x
                        and x + d <= rand_x + sqr_dim
                        and y >= rand_y
                        and y + d <= rand_y + sqr_dim
                    )
                    if overlap or new_subset or existing_subset:
                        valid = False
                        break
                if valid:
                    squares.append((rand_x, rand_y, sqr_dim))
                    matrix[rand_x : rand_x + sqr_dim, rand_y : rand_y + sqr_dim] = 1
                    placed = True
                attempts += 1
        X.append(matrix)
        Y.append(len(squares))  # Actual number of placed squares

    return np.array(X), np.array(Y)
