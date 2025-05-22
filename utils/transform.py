import open3d as o3d

# Convierte un archivo .off en un archivo .pcd
def convert_off_to_pcd(off_obj, pcd_obj, sample_size):
  mesh = o3d.io.read_triangle_mesh(off_obj)
  sample = mesh.sample_points_poisson_disk(sample_size)
  o3d.io.write_point_cloud(pcd_obj, sample)