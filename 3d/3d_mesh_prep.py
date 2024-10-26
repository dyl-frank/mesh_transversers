import argparse
import gmsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Import 3D collection for polygons
import numpy as np
import pickle as pkl

cells = {}

class Cell:
    def __init__(self, element_id, node_tags, node_coords, face_tags):
        self.element_id = element_id
        self.node_tags = node_tags
        self.node_coords = node_coords
        self.face_tags = face_tags
        self.boundary = False
        self.neighbors = {}
        self.boundary_faces = []
        self.normals = {}
        self.centroid = self.compute_centroid()
        self.solved = False
        self.solved_iter = None
        self.degree = 4

    def compute_centroid(self):
        # Calculate the centroid as the average of the node coordinates
        centroid = np.mean(self.node_coords, axis=0)
        return centroid

    def check_boundary(self):
        for face in self.face_tags:
            if global_faces.count(face) == 1:
                self.boundary = True
                self.boundary_faces.append(face)

            self.normals[face] = self.compute_normal(face)

    def compute_normal(self, face):
        # Get the node indices for the face
        p1_index = np.where(self.node_tags == face[0])[0][0]
        p2_index = np.where(self.node_tags == face[1])[0][0]
        p3_index = np.where(self.node_tags == face[2])[0][0]

        p1 = np.array(self.node_coords[p1_index])
        p2 = np.array(self.node_coords[p2_index])
        p3 = np.array(self.node_coords[p3_index])

        # Calculate two edge vectors
        vector1 = p2 - p1
        vector2 = p3 - p1

        # Compute the normal vector using the cross product
        normal = np.cross(vector1, vector2)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

        # Check if the normal needs to be inverted
        midpoint = (p1 + p2 + p3) / 3
        if np.dot(normal, (midpoint - self.centroid)) < 0:
            normal = -normal  # Flip the normal direction if needed

        return normal

    def find_neighbors(self):
        for cell in cells.values():
            if cell.element_id == self.element_id:
                continue

            # Find shared faces
            shared_faces = set(self.face_tags).intersection(cell.face_tags)
            if shared_faces:
                for face in shared_faces:
                    self.neighbors[face] = cell.element_id

    def clean(self):
        self.solved = False
        self.solved_iter = None
        self.degree = 4
        return

# Command line interface
def main(mesh_name, plot):
    global global_faces
    global_faces = []

    # Initialize gmsh API
    gmsh.initialize()

    # Read the model
    gmsh.open(mesh_name)

    # Get nodes from the mesh
    node_tags, global_node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = [global_node_coords[i:i + 3] for i in range(0, len(global_node_coords), 3)]  # 3D coordinates

    # Retrieve elements from the mesh (Element '4' is tetrahedral)
    element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()
        
    
    for elem_type, elem_tags, elem_node_tags in zip(element_types, element_tags, node_tags_per_element):
        if elem_type == 4:  # Tetrahedral elements
            tetrahedra = [elem_node_tags[i:i + 4] for i in range(0, len(elem_node_tags), 4)]
            for i, tetrahedron in enumerate(tetrahedra):
                tetra_coords = [nodes[int(tag - 1)] for tag in tetrahedron]  # Node tags start from 1
                
                faces = [
                    tuple(sorted((tetrahedron[0], tetrahedron[1], tetrahedron[2]))),
                    tuple(sorted((tetrahedron[0], tetrahedron[1], tetrahedron[3]))),
                    tuple(sorted((tetrahedron[0], tetrahedron[2], tetrahedron[3]))),
                    tuple(sorted((tetrahedron[1], tetrahedron[2], tetrahedron[3])))
                ]
                
                global_faces.extend(faces)
                cells[elem_tags[i]] = Cell(elem_tags[i], tetrahedron, tetra_coords, faces)

    # Finalize Gmsh
    gmsh.finalize()

    # Check boundaries and neighbors
    for cell in cells.values():
        cell.check_boundary()
        cell.find_neighbors()

    # Save the Cells dictionary in a binary file
    with open(f"{mesh_name[:-4]}.pkl", 'wb') as f:
        pkl.dump(cells, f)

    # Plot the tetrahedral mesh and normals if the plot flag is set
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for cell in cells.values():
            if cell.element_id == 179:
                # Prepare the vertices for the tetrahedron
                vertices = np.array(cell.node_coords)
                faces = [[vertices[j] for j in range(4) if j != i] for i in range(4)]  # Get all triangular faces

                # Create a 3D polygon collection for the tetrahedron
                poly3d = Poly3DCollection(faces, alpha=0.5, color='red' if cell.element_id==179 else 'white', edgecolor='black')
                ax.add_collection3d(poly3d)

                # Compute the centroid and add the element ID
                centroid_x, centroid_y, centroid_z = cell.centroid
                ax.text(centroid_x, centroid_y, centroid_z, str(cell.element_id), color='blue', fontsize=8, ha='center', va='center')

                # Plot the normals for boundary faces
                for face in cell.face_tags:
                    # Get the normal for the face
                    normal = cell.normals[face]

                    # Calculate the center of the face
                    p1_index = np.where(cell.node_tags == face[0])[0][0]
                    p2_index = np.where(cell.node_tags == face[1])[0][0]
                    p3_index = np.where(cell.node_tags == face[2])[0][0]

                    face_center = (np.array(cell.node_coords[p1_index]) + np.array(cell.node_coords[p2_index]) + np.array(cell.node_coords[p3_index])) / 3
                    
                    # Scale the normal for better visualization (e.g., by a factor of 0.1)
                    normal_length = 0.1  # Adjust this value for quiver length
                    ax.quiver(face_center[0], face_center[1], face_center[2],
                            normal[0] * normal_length, normal[1] * normal_length, normal[2] * normal_length,
                            color='black')

        # Set labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Tetrahedral Mesh with Element IDs and Normals as Quivers')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GMSH mesh file.")
    parser.add_argument("filename", type=str, help="Path to the GMSH mesh file (.msh)")
    parser.add_argument("--plot", action="store_true", help="Plot the mesh and normals")

    args = parser.parse_args()
    
    main(args.filename, args.plot)
