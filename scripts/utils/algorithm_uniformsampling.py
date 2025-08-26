import numpy as np
import matplotlib.pyplot as plt
import sobol_seq
from mpl_toolkits.mplot3d import Axes3D

def is_perfect_cube(n):
    cube_root = int(round(n ** (1/3.0)))
    return cube_root ** 3 == n

def generate_cubic_grid_points(N, a, b, c):
    num_points_per_axis = int(round(N ** (1/3.0)))
    
    xs = np.linspace(-a, a, num_points_per_axis)
    ys = np.linspace(-b, b, num_points_per_axis)
    zs = np.linspace(-c, c, num_points_per_axis)
    
    points = [(x, y, z) for x in xs for y in ys for z in zs]
    
    points = points[:N]
    
    return np.array(points)  # Convert to NumPy array

def generate_sobol_points(N, a, b, c):
    points = sobol_seq.i4_sobol_generate(3, N)
    
    points[:, 0] = points[:, 0] * (2 * a) - a
    points[:, 1] = points[:, 1] * (2 * b) - b
    points[:, 2] = points[:, 2] * (2 * c) - c
    
    return np.array(points)  # Convert to NumPy array

def plot_points(points, a, b, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, c='b')  # Blue points
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim([-a, a])
    ax.set_ylim([-b, b])
    ax.set_zlim([-c, c])
    plt.show()

def main():
    N = int(input("Enter the number of points: "))
    a = float(input("Enter the length a of the cube: "))
    b = float(input("Enter the length b of the cube: "))
    c = float(input("Enter the length c of the cube: "))
    
    if is_perfect_cube(N):
        points = generate_cubic_grid_points(N, a, b, c)
    else:
        points = generate_sobol_points(N, a, b, c)
    
    print("Generated {} points.".format(len(points)))
    
    plot_points(points, a, b, c)

if __name__ == "__main__":
    main()
