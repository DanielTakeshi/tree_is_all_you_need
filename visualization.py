import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D


def print_loss_by_tree(base, val):
    """
    generates a scatter plot with base displacement and prediction error as its dimenstion. Each point is one push prediction
    """
    zipped = list(zip(base, val))
    zipped.sort(key=first_value)

    base = [x for (x,y) in zipped]
    val = [y for (x,y) in zipped]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(base, val, s=4)
    ax.plot(base, base, color="orange")
    ax.set_xlabel("base loss")
    ax.set_ylabel("validation loss")
    #display(fig)
    plt.savefig(results_path+"base_vs_loss_by_tree")
    plt.close()
    clear_output(wait=True)


# Possibly from: https://stackoverflow.com/questions/22867620/
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax


def visualize_graph(X, Y, X_0, edge_index, force_node, force):
    force = force.detach().cpu().numpy()

    force_vector = force[force_node]/np.linalg.norm(force[force_node])/2
    force_A = X_0.detach().cpu().numpy()[force_node]
    force_B = X_0.detach().cpu().numpy()[force_node] + force_vector*2


    x_0 = []
    x_edges = []
    y_edges = []
    for edge in edge_index.T:
        x_0.append([X_0[edge[0]].detach().cpu().numpy(), X_0[edge[1]].detach().cpu().numpy()])
        x_edges.append([X[edge[0]].detach().cpu().numpy(), X[edge[1]].detach().cpu().numpy()])
        y_edges.append([Y[edge[0]].detach().cpu().numpy(), Y[edge[1]].detach().cpu().numpy()])
    x_0 = np.array(x_0)
    x_edges = np.array(x_edges)
    y_edges = np.array(y_edges)


    ax = plt.figure().add_subplot(projection='3d')
    fn = X_0[force_node].detach().cpu().numpy()
    ax.scatter(fn[0], fn[1], fn[2], c='m', s=50)
    x0_lc = Line3DCollection(x_0, colors=[0,0,1,1], linewidths=1)
    x_lc = Line3DCollection(x_edges, colors=[1,0,0,1], linewidths=5)
    y_lc = Line3DCollection(y_edges, colors=[0,1,0,1], linewidths=5)
    ax.add_collection3d(x0_lc)
    ax.add_collection3d(x_lc)
    ax.add_collection3d(y_lc)

    arrow_prop_dict = dict(mutation_scale=30, arrowstyle='-|>', color='m', shrinkA=0, shrinkB=0)
    a = Arrow3D([force_A[0], force_B[0]],
                [force_A[1], force_B[1]],
                [force_A[2], force_B[2]], **arrow_prop_dict)
    ax.add_artist(a)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([0, 2.0])

    custom_lines = [Line2D([0], [0], color=[0,0,1,1], lw=2),
                    Line2D([0], [0], color=[1,0,0,1], lw=4),
                    Line2D([0], [0], color=[0,1,0,1], lw=4)]

    ax.legend(custom_lines, ['Input', 'Predicted', 'GT'])


    ax = set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


def make_gif(X, Y, X_0, edge_index, force_node, force, id):
    """Make an animation. Why is it not working now?

    I'm getting confused about `edge_index`, why is it not what I'd expect?

    https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

    # This will work for a scatter plot, but here we want connected lines.
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x_edges[:,:,0], x_edges[:,:,1], x_edges[:,:,2], color='b')
    ax.scatter(y_edges[:,:,0], y_edges[:,:,1], y_edges[:,:,2], color='r')

    Parameters
    ----------
    X: the predictions from the trained model.
    Y: the ground-truth node positions (root-normalized).
    X_0: the initial node positions (root-normalized).
    force_node: node index to which force was applied.
    force: (N,3)-shaped force array, only row at index `force_node` should be nonzero.
    id: integer ID value, for the current batch item (note: batch size is 1).
    """
    force = force.detach().cpu().numpy()

    # Create a force vector. TODO(daniel): this is normalized but can we downscale?
    force_vector = force[force_node] / np.linalg.norm(force[force_node]) / 2  # why div 2 ?
    force_A = X_0.detach().cpu().numpy()[force_node]
    force_B = X_0.detach().cpu().numpy()[force_node] + force_vector * 1.0  # used to be 2.0

    # NOTE(daniel): extract edge structure of the tree before and after force.
    x_0 = []
    x_edges = []
    y_edges = []
    for edge in edge_index.T:
        x_0.append([X_0[edge[0]].detach().cpu().numpy(), X_0[edge[1]].detach().cpu().numpy()])
        x_edges.append([X[edge[0]].detach().cpu().numpy(), X[edge[1]].detach().cpu().numpy()])
        y_edges.append([Y[edge[0]].detach().cpu().numpy(), Y[edge[1]].detach().cpu().numpy()])
    x_0 = np.array(x_0)

    # x_edges, y_edges are (num_edges, 2, 3) containing positions of their vertices.
    # TODO(daniel): why are there more edges than I'd expect?
    x_edges = np.array(x_edges)
    y_edges = np.array(y_edges)

    # Debugging.
    print(X_0)
    print(edge_index)
    print(x_0.shape)
    print(x_edges.shape)
    print(y_edges.shape)

    fig = plt.figure()

    # NOTE(daniel): showing up, finally.
    #ax = Axes3D(fig)
    # https://stackoverflow.com/questions/67095247/gca-and-latest-version-of-matplotlib
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    #fn = X_0[force_node].detach().cpu().numpy()  # causes axes to be crazy?
    #ax.scatter(fn[0], fn[1], fn[2], c='m', s=50) # causes axes to be crazy?
    x0_lc = Line3DCollection(x_0,     colors=[0,0,1,1], linewidths=1)
    x_lc  = Line3DCollection(x_edges, colors=[1,0,0,1], linewidths=5)
    y_lc  = Line3DCollection(y_edges, colors=[0,1,0,1], linewidths=5)
    ax.add_collection3d(x0_lc)
    ax.add_collection3d(x_lc)
    ax.add_collection3d(y_lc)

    # Must show the force as an arrow.
    arrow_prop_dict = dict(mutation_scale=30, arrowstyle='-|>', color='m', shrinkA=0, shrinkB=0)
    a = Arrow3D([force_A[0], force_B[0]],
                [force_A[1], force_B[1]],
                [force_A[2], force_B[2]], **arrow_prop_dict)
    ax.add_artist(a)

    # NOTE(daniel): these seem to be the ranges used in their GitHub animation.
    # But I am seeing that the scale is quite off, is it due to normalization?
    # The root is set at (0,0,0), maybe let's just start from there?
    # We should dynamically adjust the ranges.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.4, 0.4])
    ax.set_zlim([0.0, 0.8])

    #custom_lines = [Line2D([0], [0], color=[0,0,1,1], lw=2),
    #                Line2D([0], [0], color=[1,0,0,1], lw=4),
    #                Line2D([0], [0], color=[0,1,0,1], lw=4)]
    #ax.legend(custom_lines, ['Input', 'GT', 'Predicted'])
    ax = set_axes_equal(ax)

    # initialization function: plot the background of each frame
    def init():
        return fig,

    # animation function.  This is called sequentially
    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim_file = f'output/{str(id).zfill(3)}.gif'
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    anim.save(anim_file, fps=30)
    print(f'Saved animation: {anim_file}')
