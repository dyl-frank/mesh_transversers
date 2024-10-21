# import pickle as pkl
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import argparse
# from matplotlib.colors import ListedColormap
# from matplotlib.cm import ScalarMappable
# import copy


# class Cell:
#     def __init__(self):
#         self.normals = {}  # Dictionary for normals to faces
#         self.neighbors = {}  # Dictionary for neighboring cells
#         self.boundary_edges = []  # List of edges that are boundaries
#         self.node_coords = []  # List of node coordinates for plotting
#         self.degree = 0  # Dependency degree
#         self.solved = False  # Whether the cell is solved
#         self.solved_iter = None  # Iteration when it was solved
#     def clean(self):
#         self.solved = False
#         self.solved_iter = None
#         self.degree = 3
#         return



# def main(filename, snorder, plot):
#     # Order of SN directions
#     # Gaussian quadrature for weights and points
#     mu_q, w_q = np.polynomial.legendre.leggauss(int(snorder))
#     omegas = []
#     for mu in mu_q:
#         omegas.append(np.array([mu, np.sqrt(1-mu**2)]))
#     # Load cells from the pickle file
#     with open(filename, 'rb') as f:
#         cells = pkl.load(f)

#     # Store the original neighbors for each cell
#     for cell in cells.values():
#         cell.original_neighbors = copy.deepcopy(cell.neighbors)

#     frames = []  # List of states to animate
#     omega_idxs = []  # List of quivers to animate

#     # Sweep through each omega direction
#     for omega in omegas:
#         # Reset the neighbors to the original state for the new omega
#         for cell in cells.values():
#             cell.neighbors = copy.deepcopy(cell.original_neighbors)
#             cell.clean()  # Reset the cell state

#         # Preliminary screening: reduce dependencies based on omega direction
#         for cell in cells.values():
#             # Reduce degree for outgoing faces
#             for normal in cell.normals.values():
#                 if np.dot(normal, omega) >= 0:
#                     cell.degree -= 1

#             # Reduce degree for incoming boundary faces
#             for edge in cell.boundary_edges:
#                 if np.dot(cell.normals[edge], omega) < 0:
#                     cell.degree -= 1

#         # Track states for animation frames
#         n_solved = 0
#         i = 0

#         while n_solved < len(cells):
#             # Solve cells whose dependencies are zero
#             for cell_id, cell in cells.items():
#                 if cell.degree == 0 and not cell.solved:
#                     cell.solved = True
#                     cell.solved_iter = i
#                     n_solved += 1

#             # Update dependencies for unsolved cells
#             for cell_id, cell in cells.items():
#                 if not cell.solved:
#                     neighbors_to_kill = []
#                     for edge, neighbor_id in cell.neighbors.items():
#                         if cells[neighbor_id].solved and np.dot(cell.normals[edge], omega) < 0:
#                             cell.degree -= 1
#                             neighbors_to_kill.append(edge)

#                     # Remove used neighbors from the temporary copy
#                     for neighbor in neighbors_to_kill:
#                         del cell.neighbors[neighbor]

#             # Append the current state to frames
#             frames.append(copy.deepcopy(cells))  # Capture state for each iteration
#             omega_idxs.append(omega)
#             i += 1
#             print("hello worlds")
#     omega_idxs.extend([omega for _ in range(i)])
#     # Convert frames to a list if it's not already
#     frames = list(frames)

#     # Plot or animate if needed
#     if plot:
#         animate_sweep(frames, omega_idxs)
#     # Order of SN directions

 


# def plot_sweep(cells, ax):
#     ax.clear()
#     for cell in cells.values():
#         x_coords = [coord[0] for coord in cell.node_coords]
#         y_coords = [coord[1] for coord in cell.node_coords]

#         # Close the polygon
#         x_coords.append(cell.node_coords[0][0])
#         y_coords.append(cell.node_coords[0][1])

#         # Plot the polygon
#         color = 'green' if cell.solved else 'white'
#         ax.fill(x_coords, y_coords, color, edgecolor='black', linewidth=1)

#         # Display the iteration number if solved
#         if cell.solved_iter is not None:
#             centroid_x = sum(x_coords[:-1]) / len(cell.node_coords)
#             centroid_y = sum(y_coords[:-1]) / len(cell.node_coords)
#             ax.text(centroid_x, centroid_y, str(cell.solved_iter), color='black', fontsize=8, ha='center', va='center')

#     # Set equal scaling and labels
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title('Triangular Mesh with Solvable Iteration')


# def animate_sweep(frames, omega_idxs):
#     fig, ax = plt.subplots()

#     def update(frame, omega_idxs = []):
#         plot_sweep(frame, ax)

#     ani = animation.FuncAnimation(fig, update, frames=frames, repeat=True, blit=False)
#     plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Mesh Traverser")
#     parser.add_argument("filename", type=str, help="Path to Processed Mesh (.pkl)")
#     parser.add_argument("snorder", type=int, help="x-component of field")
#     parser.add_argument("--plot", action="store_true", help="Make an Animation of the Traversal")

#     args = parser.parse_args()
#     main(args.filename, args.snorder, args.plot)

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import copy


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
        self.degree = 3
        return


def main(filename, snorder, plot):
    # Order of SN directions
    # Gaussian quadrature for weights and points
    mu_q, w_q = np.polynomial.legendre.leggauss(int(snorder))
    omegas = []
    for mu in mu_q:
        omegas.append(np.array([mu, np.sqrt(1 - mu**2)]))
    # Load cells from the pickle file
    with open(filename, 'rb') as f:
        cells = pkl.load(f)

    # Store the original neighbors for each cell
    for cell in cells.values():
        cell.original_neighbors = copy.deepcopy(cell.neighbors)

    frames = []  # List of states to animate
    omega_idxs = []  # List of quivers to animate
    mu_idxs = []  # List of mus to print

    # Sweep through each omega direction
    for omega, mu in zip(omegas, mu_q):
        # Reset the neighbors to the original state for the new omega
        for cell in cells.values():
            cell.neighbors = copy.deepcopy(cell.original_neighbors)
            cell.clean()  # Reset the cell state

        # Preliminary screening: reduce dependencies based on omega direction
        for cell in cells.values():
            # Reduce degree for outgoing faces
            for normal in cell.normals.values():
                if np.dot(normal, omega) >= 0:
                    cell.degree -= 1

            # Reduce degree for incoming boundary faces
            for edge in cell.boundary_edges:
                if np.dot(cell.normals[edge], omega) < 0:
                    cell.degree -= 1

        # Track states for animation frames
        n_solved = 0
        i = 0

        while n_solved < len(cells):
            # Solve cells whose dependencies are zero
            for cell_id, cell in cells.items():
                if cell.degree == 0 and not cell.solved:
                    cell.solved = True
                    cell.solved_iter = i
                    n_solved += 1

            # Update dependencies for unsolved cells
            for cell_id, cell in cells.items():
                if not cell.solved:
                    neighbors_to_kill = []
                    for edge, neighbor_id in cell.neighbors.items():
                        if cells[neighbor_id].solved and np.dot(cell.normals[edge], omega) < 0:
                            cell.degree -= 1
                            neighbors_to_kill.append(edge)

                    # Remove used neighbors from the temporary copy
                    for neighbor in neighbors_to_kill:
                        del cell.neighbors[neighbor]

            # Append the current state to frames
            frames.append(copy.deepcopy(cells))  # Capture state for each iteration
            omega_idxs.append(omega)
            mu_idxs.append(mu)
            i += 1

    # Convert frames to a list if it's not already
    frames = list(frames)

    # Plot or animate if needed
    if plot:
        animate_sweep(frames, omega_idxs, mu_idxs)


def plot_sweep(cells, ax, omega, mu):
    ax.clear()
    for cell in cells.values():
        x_coords = [coord[0] for coord in cell.node_coords]
        y_coords = [coord[1] for coord in cell.node_coords]

        # Close the polygon
        x_coords.append(cell.node_coords[0][0])
        y_coords.append(cell.node_coords[0][1])

        # Plot the polygon
        color = 'green' if cell.solved else 'white'
        ax.fill(x_coords, y_coords, color, edgecolor='black', linewidth=1)

        # Display the iteration number if solved
        if cell.solved_iter is not None:
            centroid_x = sum(x_coords[:-1]) / len(cell.node_coords)
            centroid_y = sum(y_coords[:-1]) / len(cell.node_coords)
            ax.text(centroid_x, centroid_y, str(cell.solved_iter), color='black', fontsize=8, ha='center', va='center')

    # Plot omega as a quiver from the origin
    # Calculate the center of the plot based on the mesh bounds
    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2

    # Plot the quiver with the tail at the center of the plot
    ax.quiver(x_center, y_center, 5*omega[0], 5*omega[1], angles='xy', scale_units='xy', scale=2, color='red', label='Omega', width=0.05)

    # Set equal scaling and labels
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Triangular Mesh with Solvable Iteration mu= {mu}')
    ax.legend()


def animate_sweep(frames, omega_idxs, mu_idxs):
    fig, ax = plt.subplots()

    def update(frame_idx):
        frame = frames[frame_idx]
        omega = omega_idxs[frame_idx]
        mu = mu_idxs[frame_idx]
        plot_sweep(frame, ax, omega, mu)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=True, blit=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Traverser")
    parser.add_argument("filename", type=str, help="Path to Processed Mesh (.pkl)")
    parser.add_argument("snorder", type=int, help="SN order for quadrature")
    parser.add_argument("--plot", action="store_true", help="Make an Animation of the Traversal")

    args = parser.parse_args()
    main(args.filename, args.snorder, args.plot)
