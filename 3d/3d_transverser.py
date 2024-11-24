import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy
import matplotlib.cm as cm


class Cell:
    def __init__(self):
        self.normals = {}             # Dictionary of face normals
        self.neighbors = {}           # Dictionary of neighboring cells
        self.boundary_edges = []      # List of boundary edges
        self.node_coords = []         # Node coordinates for plotting
        self.degree = 0               # Dependency degree
        self.solved = False           # Solved state
        self.solved_iter = None       # Iteration when solved

    def reset(self):
        """Reset cell state for a new sweep."""
        self.solved = False
        self.solved_iter = None
        self.degree = 4


def calculate_omegas(snorder):
    """Calculate spherical coordinate vectors for omega directions."""
    # Polar angles and weights
    mu_q, w_q = np.polynomial.legendre.leggauss(snorder)  # Polar angles
    phi_q = np.linspace(0, np.pi/2, snorder)  # Azimuthal angles


    omegas = []
    for phi in phi_q:
        for mu in mu_q:
            x = np.sqrt(1 - mu**2) * np.cos(phi)
            y = np.sqrt(1 - mu**2) * np.sin(phi)
            z = mu
            omegas.extend([np.array([x, y, z]), np.array([-x, y, z]), np.array([-x, -y, z]), np.array([x, -y, z])])
    # print(omegas)
    return omegas, mu_q, phi_q


def load_cells(filename):
    """Load cell data from a pickle file."""
    with open(filename, 'rb') as f:
        cells = pkl.load(f)
    for cell in cells.values():
        cell.original_neighbors = copy.deepcopy(cell.neighbors)
    return cells


def sweep_omegas(cells, omegas):
    frames, omega_idxs= [], []

    for omega in omegas:
        for cell in cells.values():
            cell.neighbors = copy.deepcopy(cell.original_neighbors)
            cell.reset()

        for cell in cells.values():
            for normal in cell.normals.values():
                if np.dot(normal, omega) >= 0:
                    cell.degree -= 1
            for face in getattr(cell, "boundary_faces", []):
                if np.dot(cell.normals[face], omega) < 0:
                    cell.degree -= 1

        n_solved, i = 0, 1
        while n_solved < len(cells):
            for cell in cells.values():
                if cell.degree <= 0 and not cell.solved:
                    cell.solved = True
                    cell.solved_iter = i
                    n_solved += 1

            for cell in cells.values():
                if not cell.solved:
                    neighbors_to_kill = [
                        face for face, neighbor_id in cell.neighbors.items()
                        if cells[neighbor_id].solved and np.dot(cell.normals[face], omega) < 0
                    ]
                    for face in neighbors_to_kill:
                        cell.degree -= 1
                        del cell.neighbors[face]

            frames.append(copy.deepcopy(cells))
            omega_idxs.append(omega)

            i += 1

    global max_iter 
    max_iter = max(cell.solved_iter for cell in cells.values() if cell.solved_iter)
    return frames, omega_idxs


def plot_sweep(cells, ax, omega):
    ax.clear()
    for cell in cells.values():
        if cell.solved:
            vertices = np.array(cell.node_coords)
            faces = [[vertices[j] for j in range(4) if j != i] for i in range(4)]
            color = cm.rainbow(cell.solved_iter / max_iter)
            poly3d = Poly3DCollection(faces, alpha=0.5, color=color, edgecolor='black')
            ax.add_collection3d(poly3d)
            centroid = vertices.mean(axis=0)
            ax.text(*centroid, str(cell.solved_iter), color='black', fontsize=8, ha='center', va='center')

    ax.quiver(-1.5, -1.5, -1.5, *omega, color='red', label='Omega', length=0.5)
    ax.set(xlabel='X', ylabel='Y', zlabel='Z',
           title=f'Sweep Ordering for\n $\Omega$ = {omega}')
    ax.legend()


def animate_sweep(frames, omega_idxs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        plot_sweep(frames[frame_idx], ax, omega_idxs[frame_idx])

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1, repeat=True)
    plt.show()


def main(filename, snorder, plot):
    cells = load_cells(filename)
    omegas, mu_q, phi_q = calculate_omegas(snorder)
    frames, omega_idxs = sweep_omegas(cells, omegas)
    if plot:
        animate_sweep(frames, omega_idxs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Traverser")
    parser.add_argument("filename", type=str, help="Path to Processed Mesh (.pkl)")
    parser.add_argument("snorder", type=int, help="SN order for quadrature")
    parser.add_argument("--plot", action="store_true", help="Make an Animation of the Traversal")
    args = parser.parse_args()
    main(args.filename, args.snorder, args.plot)
