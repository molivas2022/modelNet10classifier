import open3d as o3d
import numpy as np

# Convierte un archivo .off en un archivo .pcd
def convert_off_to_pcd(off_obj, pcd_obj, sample_size):
  mesh = o3d.io.read_triangle_mesh(off_obj)
  sample = mesh.sample_points_poisson_disk(sample_size)
  o3d.io.write_point_cloud(pcd_obj, sample)

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points
