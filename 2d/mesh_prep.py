import argparse
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

cells = {}

class Cell:
    def __init__(self, element_id, node_tags, node_coords, edge_tags):
        self.element_id = element_id
        self.node_tags = node_tags
        self.node_coords = node_coords
        self.edge_tags = edge_tags
        self.boundary = False
        self.neighbors = {}
        self.boundary_edges = []
        self.boundary_normals = {}
        self.normals = {}
        self.centroid = self.compute_centroid()  # Calculate the centroid upon initialization
        self.solved = False
        self.solved_iter = None
        self.degree = 3
        
    def compute_centroid(self):
        # Calculate the centroid as the average of the node coordinates
        centroid = np.mean(self.node_coords, axis=0)
        return centroid

    def check_boundary(self):
        for edge in self.edge_tags:
            if global_edges.count(edge) == 1:
                self.boundary = True
                self.boundary_edges.append(edge)

            
            self.normals[edge] = self.compute_normal(edge)
    
    def compute_normal(self, edge):
        p1_index = np.where(self.node_tags == edge[0])[0][0]
        p2_index = np.where(self.node_tags == edge[1])[0][0]
        
        p1 = np.array(self.node_coords[p1_index])
        p2 = np.array(self.node_coords[p2_index])

        # Calculate the edge vector from p1 to p2
        edge_vector = p2 - p1  # This gives the direction from p1 to p2

        # Get the normal vector by rotating 90 degrees (for 2D)
        normal = np.array([edge_vector[1], -edge_vector[0]])  # Rotate 90 degrees counter-clockwise for outward normal
        
        # Check if the normal needs to be inverted
        midpoint = (p2 + p1) / 2
        if np.dot(normal, (midpoint - self.centroid)) < 0:
            normal = -normal  # Flip the normal direction if needed

        return normal

    def find_neighbors(self):

        for cell in cells.values():
            # Skip if it's the same cell
            if cell.element_id == self.element_id:
                continue

            # Find shared edges
            shared_edges = set(self.edge_tags).intersection(cell.edge_tags)
            if shared_edges:
                # For each shared edge, store the neighbor
                for edge in shared_edges:
                    self.neighbors[edge] = cell.element_id

    def clean(self):
        self.solved = False
        self.solved_iter = None
        self.degree = 3
        return

                           
# Command line interface
def main(mesh_name, plot):
    global global_edges
    global_edges = []

    # Initialize gmsh API
    gmsh.initialize()

    # Read the model
    gmsh.open(mesh_name)

    # Get nodes from the mesh
    node_tags, global_node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = [(global_node_coords[i], global_node_coords[i + 1]) for i in range(0, len(global_node_coords), 3)]  # Use 2D coordinates

    # Retrieve elements from the mesh (Element '2' is triangular)
    element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()

    # Loop through each element type
    for elem_type, elem_tags, elem_node_tags in zip(element_types, element_tags, node_tags_per_element):
        if elem_type == 2:  # Triangular elements
            triangles = [elem_node_tags[i:i + 3] for i in range(0, len(elem_node_tags), 3)]
            for i, triangle in enumerate(triangles):
                triangle_coords = [nodes[tag - 1] for tag in triangle]  # Node tags start from 1
                
                edges = [tuple(sorted((triangle[0], triangle[1]))),
                         tuple(sorted((triangle[1], triangle[2]))),
                         tuple(sorted((triangle[2], triangle[0])))]
                
                global_edges.extend(edges)
                cells[elem_tags[i]] = Cell(elem_tags[i], triangle, triangle_coords, edges)

    # Finalize Gmsh
    gmsh.finalize()

    # Check boundaries and neighbors
    for cell in cells.values():
        cell.check_boundary()
        cell.find_neighbors()

    # Save the Cells dictionary in a binary file
    with open(f"{mesh_name[:-4]}.pkl", 'wb') as f:
        pkl.dump(cells, f)

    # Plot the triangular mesh and normals if the plot flag is set
    if plot:
        for cell in cells.values():
            x_coords = [coord[0] for coord in cell.node_coords]
            y_coords = [coord[1] for coord in cell.node_coords]

            # Close the triangle
            x_coords.append(cell.node_coords[0][0])
            y_coords.append(cell.node_coords[0][1])

            # Plot the triangle
            color = 'red' if cell.boundary else 'white'
            plt.fill(x_coords, y_coords, color, edgecolor='black', linewidth=1)

            # Compute the centroid and add the element ID
            centroid_x = sum(x_coords[:-1]) / 3
            centroid_y = sum(y_coords[:-1]) / 3
            plt.text(centroid_x, centroid_y, str(cell.element_id), color='blue', fontsize=8, ha='center', va='center')

            # Plot the normals for boundary edges
            
            for edge in cell.edge_tags:
                # Get the normal for the edge
                normal = cell.normals[edge]
                
                # Calculate the center of the edge
                p1_index = np.where(cell.node_tags == edge[0])[0][0]
                p2_index = np.where(cell.node_tags == edge[1])[0][0]
                
                edge_center = (np.array(cell.node_coords[p1_index]) + np.array(cell.node_coords[p2_index])) / 2
                
                # Scale the normal for better visualization (e.g., by a factor of 0.1)
                normal_length = 0.1  # Adjust this value for quiver length
                plt.quiver(edge_center[0], edge_center[1], normal[0] * normal_length, normal[1] * normal_length, 
                           angles='xy', scale_units='xy', scale=1, color='black')

        # Set equal scaling and display the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Triangular Mesh with Element IDs and Normals as Quivers')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GMSH mesh file.")
    parser.add_argument("filename", type=str, help="Path to the GMSH mesh file (.msh)")
    parser.add_argument("--plot", action="store_true", help="Plot the mesh and normals")

    args = parser.parse_args()
    
    main(args.filename, args.plot)
