import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

# Visualiza en un notebook un archivo .pcd
def notebook_plot_pcd(pcd_obj: str):
    pcd = o3d.io.read_point_cloud(pcd_obj)
    points = np.asarray(pcd.points)

    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    elif pcd.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(pcd.normals) * 0.5

    fig = go.Figure(
    data=[
        go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(size=1, color=colors)
    )
    ],
    layout=dict(
        scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
    )
    )
    fig.show()