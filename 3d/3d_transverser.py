import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy
import time
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class Cell:
    def __init__(self):
        self.normals = {}  # Dictionary for normals to faces
        self.neighbors = {}  # Dictionary for neighboring cells
        self.boundary_edges = []  # List of edges that are boundaries
        self.node_coords = []  # List of node coordinates for plotting
        self.degree = 0  # Dependency degree
        self.solved = False  # Whether the cell is solved
        self.solved_iter = None  # Iteration when it was solved

    def clean(self):
        self.solved = False
        self.solved_iter = None
        self.degree = 4
        return


def main(filename, snorder, plot):
    # Get Gaussian quadrature points and weights
    mu_q, w_q = np.polynomial.legendre.leggauss(int(snorder))  # Zenith angles

    # Calculate the azimuthal angles
    phi_q = [2 * np.pi / snorder * i for i in range(snorder)]

    # Initialize omegas list
    omegas = []

    # Create the omega vectors in spherical coordinates
    for phi in phi_q:
        for mu in mu_q:
            # Convert spherical to Cartesian coordinates
            x = np.sqrt(1 - mu**2) * np.cos(phi)  # x-component
            y = np.sqrt(1 - mu**2) * np.sin(phi)  # y-component
            z = mu  # z-component
            omegas.append(np.array([x, y, z]))  # Store the vector

    # Load cells from the pickle file
    with open(filename, 'rb') as f:
        cells = pkl.load(f)

    # Store the original neighbors for each cell
    for cell in cells.values():
        cell.original_neighbors = copy.deepcopy(cell.neighbors)

    frames = []  # List of states to animate
    omega_idxs = []  # List of quivers to animate
    mu_idxs = []  # List of mus to print
    phi_idxs = []  # List of phis to print

    # Sweep through each omega direction
    for omega, mu,phi in zip(omegas, mu_q, phi_q):
        # Reset the neighbors to the original state for the new omega
        for cell in cells.values():
            cell.neighbors = copy.deepcopy(cell.original_neighbors)
            cell.clean()  # Reset the cell state

        # Preliminary screening: reduce dependencies based on omega direction
        dg_0  =cell.degree
        for cell in cells.values():
            # Reduce degree for outgoing faces
            for normal in cell.normals.values():
                if np.dot(normal, omega) >= 0:
                    cell.degree -= 1
                    print("killing boundary incoming", cell.element_id, dg_0, cell.degree)

            # Reduce degree for incoming boundary faces
            dg_0  =cell.degree
            for face in cell.boundary_faces:
                if np.dot(cell.normals[face], omega) < 0:
                    cell.degree -= 1
                    print("killing neighbors", cell.element_id, dg_0, cell.degree)
                    # time.sleep(1)
        # Track states for animation frames
        n_solved = 0
        i = 1

        while n_solved < len(cells):
            # Solve cells whose dependencies are zero
            for cell_id, cell in cells.items():
                if cell.degree <= 0 and not cell.solved:
                    cell.solved = True
                    cell.solved_iter = i
                    n_solved += 1
                    print(cell.degree)

            # Update dependencies for unsolved cells
            for cell_id, cell in cells.items():
                # if cell.degree < 0:
                #     # print(cell.degree)

                if not cell.solved:
                    neighbors_to_kill = []
                    for face, neighbor_id in cell.neighbors.items():
                        if cells[neighbor_id].solved and np.dot(cell.normals[face], omega) < 0:
                            cell.degree -= 1
                            neighbors_to_kill.append(face)

                    # Remove used neighbors from the temporary copy
                    for neighbors_face in neighbors_to_kill:
                        del cell.neighbors[neighbors_face]

            # Append the current state to frames
            frames.append(copy.deepcopy(cells))  # Capture state for each iteration
            omega_idxs.append(omega)
            mu_idxs.append(mu)
            phi_idxs.append(phi)
            i += 1
        # print("here")
    # Convert frames to a list if it's not already
    frames = list(frames)

    global max_iter 
    max_iter = max(cell.solved_iter for cell in cells.values() if cell.solved_iter is not None)

    # Plot or animate if needed
    if plot:
        animate_sweep(frames, omega_idxs, mu_idxs, phi_idxs)


def plot_sweep(cells, ax, omega, mu, phi):
    ax.clear()

    # Get the maximum solved_iter to normalize the color mapping
    # max_iter = max(cell.solved_iter for cell in cells.values() if cell.solved_iter is not None)

    for cell in cells.values():
        # Prepare the vertices for the tetrahedron
        vertices = np.array(cell.node_coords)

        # Create faces for the tetrahedron
        faces = [
            [vertices[j] for j in range(4) if j != i] for i in range(4)
        ]

        # Determine color based on solved_iter
        if cell.solved_iter is not None:
            # Normalize the solved_iter to [0, 1] range
            norm_value = cell.solved_iter / max_iter
            # Get the color from the rainbow colormap
            color = cm.rainbow(norm_value)
        else:
            # Default color for unsolved cells
            color = 'white'

        # Create a 3D polygon collection for the tetrahedron
        poly3d = Poly3DCollection(faces, alpha=0.5, color=color, edgecolor='black')
        ax.add_collection3d(poly3d)

        # Display the iteration number if solved
        if cell.solved_iter is not None:
            centroid = np.mean(vertices, axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], str(cell.solved_iter), color='black', fontsize=8, ha='center', va='center')

    # Plot the quiver with the tail at the origin
    ax.quiver(-1.5, -1.5, -1.5, omega[0], omega[1], omega[2], color='red', label='Omega', length=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Tetrahedral Mesh with Solvable\n Iteration mu= {mu}\n phi={phi*180/np.pi}')
    ax.legend()



    # Plot the quiver with the tail at the origin
    ax.quiver(-1.5, -1.5, -1.5, omega[0], omega[1], omega[2], color='red', label='Omega', length=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Tetrahedral Mesh with Solvable\n Iteration mu= {mu}\n phi={phi*180/np.pi}')
    ax.legend()


def animate_sweep(frames, omega_idxs, mu_idxs, phi_idxs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        frame = frames[frame_idx]
        omega = omega_idxs[frame_idx]
        mu = mu_idxs[frame_idx]
        phi = phi_idxs[frame_idx]
        plot_sweep(frame, ax, omega, mu, phi)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=True, blit=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Traverser")
    parser.add_argument("filename", type=str, help="Path to Processed Mesh (.pkl)")
    parser.add_argument("snorder", type=int, help="SN order for quadrature")
    parser.add_argument("--plot", action="store_true", help="Make an Animation of the Traversal")

    args = parser.parse_args()
    main(args.filename, args.snorder, args.plot)
