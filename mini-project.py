import cv2
import numpy as np
import open3d as o3d
from skimage import measure

def generate_point_cloud(image_path, scale=1.0):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Resize for consistency
    
    # Generate point cloud from height map
    h, w = img.shape
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    z = img / 255.0 * scale  # Normalize and scale height
    
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    return points

def create_mesh_from_point_cloud(points):
    # Convert point cloud to volumetric grid
    voxel_size = 0.05  # Define voxel resolution
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = ((max_bound - min_bound) / voxel_size).astype(int)
    grid = np.zeros(grid_shape)
    
    # Populate the voxel grid
    for point in points:
        index = ((point - min_bound) / voxel_size).astype(int)
        index = np.clip(index, 0, np.array(grid.shape) - 1)  # Ensure indices stay within bounds
        grid[tuple(index)] = 1  # Mark as occupied
    
    # Apply marching cubes to extract mesh
    verts, faces, _, _ = measure.marching_cubes(grid, level=0.5)
    verts = verts * voxel_size + min_bound  # Scale vertices back
    
    return verts, faces

def visualize_mesh(verts, faces):
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    o3d.visualization.draw_geometries([mesh])

# Example usage
image_path = 'c:/Users/admin/Downloads/shoe.png' 
points = generate_point_cloud(image_path, scale=2.0)
verts, faces = create_mesh_from_point_cloud(points)
visualize_mesh(verts, faces)
