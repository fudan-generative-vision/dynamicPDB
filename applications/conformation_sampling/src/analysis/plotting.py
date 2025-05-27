import numpy as np
from data import utils as du
from analysis import utils as au

from matplotlib.animation import FuncAnimation
from matplotlib import animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import List
import geomstats.visualization as visualization
import matplotlib.cm as cm


def create_scatter(pos_3d: np.ndarray,
                   mode: str = 'markers',
                   marker_size: int = None,
                   name: str = None,
                   opacity: float = None,
                   color: List = None,
                   colorscale: str = None,
                   ):
    """Creates Scatter3D objects for use in plotly.

    Args:
        pos_3d: [N, 3] array containing N points with
            euclidean coordinates.
        mode: How to display points.
            Use 'markers' for scatter.
            Use 'lines' for lines connecting consecutive points.
            Use 'lines+markers' for scatter and lines.
        marker_size: Size of markers.
        name: Label of plotting layer to be displayed in legend.
        opacity: Transparency of points.

    Example use:
    > sample_bb_3d = plotting.create_scatter(
    > ca_pos, mode='lines+markers', marker_size=3, opacity=1.0)
        plotting.plot_traces([sample_bb_3d])
    """
    x, y, z = np.split(pos_3d, 3, axis=-1)
    args_dict = {
        'x': x[:, 0],
        'y': y[:, 0],
        'z': z[:, 0],
        'mode': mode,
        'marker': {}
    }
    if marker_size:
        args_dict['marker']['size'] = marker_size
    if name:
        args_dict['name'] = name
    if opacity:
        args_dict['opacity'] = opacity
    if color:
        args_dict['marker']['color'] = color
    if colorscale:
        args_dict['marker']['colorscale'] = colorscale
    return go.Scatter3d(**args_dict)


def create_cones(
        tail_3d: np.ndarray,
        head_3d: np.ndarray,
        sizemode: str = 'absolute',
        name: str = None,
        opacity: float = None,
        sizeref: int = None,
    ):
    """Creates Cone object for use in plotly.

    Args:
        tail_3d: [N, 3] array containing N points for where the cones begin.
        head_3d: [N, 3] array containing N points of the offsets from tail_3d.
        sizemode: Determines size the cones.
        sizeref: Scaling factor for cone size.
        name: Label of plotting layer to be displayed in legend.
        opacity: Transparency of points.
    """
    x, y, z = np.split(tail_3d, 3, axis=-1)
    u, v, w = np.split(head_3d, 3, axis=-1)
    args_dict = {
        'x': x[:, 0],
        'y': y[:, 0],
        'z': z[:, 0],
        'u': u[:, 0],
        'v': v[:, 0],
        'w': w[:, 0]
    }
    if sizemode:
        args_dict['sizemode'] = sizemode
    if name:
        args_dict['name'] = name
    if opacity:
        args_dict['opacity'] = opacity
    if sizeref:
        args_dict['sizeref'] = sizeref
    return go.Cone(**args_dict)


def plot_traces(
        fig_traces,
        height=500,
        width=600,
        title=None
    ):
    """Constructs
    """
    layout_args = {
        'height': height,
        'width': width,
    }
    if title is not None:
        layout_args['title'] = title
    layout = go.Layout(**layout_args)
    fig = go.Figure(data=fig_traces, layout=layout)
    fig.show()


def plot_traces_and_layout(
        fig_traces,
        layout
    ):
    fig = go.Figure(data=fig_traces, layout=layout)
    fig.show()


def create_static_layout(
        height=500,
        width=600,
        title=None,
        xaxis=None,
        yaxis=None,
    ):
    layout_args = {
        'height': height,
        'width': width,
    }
    if title is not None:
        layout_args['title'] = title
    if xaxis is not None:
        layout_args['xaxis'] = dict(range=xaxis, autorange=False)
    if yaxis is not None:
        layout_args['yaxis'] = dict(range=yaxis, autorange=False)
    return go.Layout(**layout_args)


def create_dynamic_layout(
        height=500,
        width=600,
        title=None,
        scene_range=None
    ):
    layout_args = {
        'height': height,
        'width': width,
        'autosize': False,
        'updatemenus': [{
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 30
                            },
                            "transition": {
                                "duration": 30
                            }
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    }
    if scene_range is not None:
        layout_args['scene'] = {
            'xaxis': scene_range,
            'yaxis': scene_range,
            'zaxis': scene_range,
            'aspectmode': 'cube',
        }
    if title is not None:
        layout_args['title'] = title
    return go.Layout(**layout_args)


# Plot samples
def plot_sample_grid(samples, num_res, motif_bb_3d=None, true_bb_3d=None):
    ncols, nrows = 3, 3
    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{'type': 'surface'}] * nrows]*ncols)
    fig.update_layout(
        title_text=f'Samples',
        height=1000,
        width=1000,
    )
    for i in range(nrows):
        for j in range(ncols):
            b_idx = i*nrows+j
            sample_ij = samples[b_idx][:(num_res[b_idx])]
            sample_bb_3d = create_scatter(
                sample_ij, mode='lines+markers', marker_size=3,
                opacity=1.0, name=f'Sample {i*nrows+j}: length_{num_res[b_idx]}')
            fig.add_trace(sample_bb_3d, row=i+1, col=j+1)

            if motif_bb_3d is not None: fig.add_trace(motif_bb_3d, row=i+1, col=j+1)
            if true_bb_3d is not None: fig.add_trace(true_bb_3d, row=i+1, col=j+1)


    fig.show()


def plot_se3(se3_vec, ax_lim=None, title=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    visualization.plot(se3_vec, ax=ax, space="SE3_GROUP")
    bb_trans = se3_vec[:, 3:]
    ln = ax.plot(bb_trans[:,0], bb_trans[:,1], bb_trans[:,2], alpha=0.4)
    if ax_lim is not None:
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)
    if title:
        ax.set_title(title)
    return ln


def viz_frames(rigids, mask, ax, scale_factor=10.0, title='', ax_lim=10.0):
    viz_mask = du.move_to_np(mask).astype(bool)
    frames = du.move_to_np(rigids)[viz_mask]
    se3_vec = au.rigids_to_se3_vec(frames, scale_factor=scale_factor)
    plot_se3(se3_vec, ax=ax, title=title, ax_lim=ax_lim)


def plt_3d(x, ax, color=None, s=20, mode='scatter', ax_lim=None):
    if x.ndim != 2:
        raise ValueError(f'Invalid dimension for x: {x.ndim} != 2')
    if x.shape[-1] != 3:
        raise ValueError(f'Final dimension of x: {x.shape[-1]} != 3')
    if mode == 'scatter':
        ax.scatter3D(
            x[:, 0],
            x[:, 1],
            x[:, 2],
            color=color,
            s=s)
    elif mode == 'line':
        ax.plot3D(
            x[:, 0],
            x[:, 1],
            x[:, 2],
            alpha=0.2)
    else:
        raise ValueError(f'Unrecognized mode: {mode}')
    if ax_lim is not None:
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)


def write_traj(
        sample_traj,
        save_path,
        res_mask=None,
        ax_lim=15.0,
        scale_factor=1.0,
        se3_vecs=True,
    ):
    fig, ax = plt.subplots(1, 1, figsize=[8, 8], subplot_kw=dict(projection="3d"))
    if res_mask is not None:
        bb_mask = du.move_to_np(res_mask).astype(bool)

    def extract_se3_vec(step):
        frame = sample_traj[step]
        if res_mask is not None:
            frame = frame[bb_mask]
        se3_vec = au.rigids_to_se3_vec(du.move_to_np(frame), scale_factor=scale_factor)
        return se3_vec

    def plot_trans(frame):
        colors = cm.brg(np.linspace(0, 1, frame.shape[0]))
        _ = plt_3d(frame[:, 4:], ax, color=colors, s=30, mode='scatter', ax_lim=ax_lim) 
        _ = plt_3d(frame[:, 4:], ax, color=colors, s=30, mode='line', ax_lim=ax_lim)

    if se3_vecs:
        _ = plot_se3(extract_se3_vec(0), ax=ax, ax_lim=ax_lim)
    else:
        _ = plot_trans(du.move_to_np(sample_traj[0]))

    def update(frame):
        ax.clear()
        if se3_vecs:
            plot_se3(extract_se3_vec(frame), ax=ax, ax_lim=ax_lim)
        else:
            plot_trans(du.move_to_np(sample_traj[frame]))
    anim = FuncAnimation(
        fig,
        update,
        frames=list(range(1, sample_traj.shape[0])),
        interval=10,
        blit=False)
    writergif = animation.PillowWriter(fps=30) 
    anim.save(save_path, writer=writergif)


if __name__ == 'main':
    pass