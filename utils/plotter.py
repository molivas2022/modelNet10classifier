import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

# Visualiza en un notebook un np.array de puntos de una pcd
def notebook_plot_pcd_from_points(points, output_size=(700, 700), zoom=1.5):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=1)
            )
        ],
        layout=dict(
            width=output_size[0],
            height=output_size[1],
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=zoom, y=zoom, z=zoom))
            )
        )
    )
    fig.show()

# Visualiza en un notebook un archivo .pcd
def notebook_plot_pcd_from_file(path, output_size=(700, 700), zoom=1.5):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    notebook_plot_pcd_from_points(points, output_size=output_size, zoom=zoom)