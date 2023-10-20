import torch
import corner
import matplotlib.pyplot as plt

def plot_loss(losses, fname=None, loss_name='KL Divergence', **kwargs):
    """ plot training and validation loss as a function of epoch """

    fig, ax = plt.subplots(**kwargs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(loss_name)

    if isinstance(losses, dict):
        nepochs = len(list(losses.values())[0])
    else:
        nepochs = len(losses)

    for key, loss in losses.items():
        ax.plot(range(nepochs), loss, label=key.capitalize())

    ax.legend()

    if fname is not None:
        fig.savefig(fname)

    return fig, ax

def plot_corner_bilby(xs, fname=None, **kwargs):
    """ Make corner plots, bilby-style. """
    
    if isinstance(xs, torch.Tensor):
        xs = xs.cpu().detach().numpy().squeeze()
    
    defaults_kwargs = dict(
        bins=50, smooth=0.9,
        title_kwargs=dict(fontsize=16),
        color='#0072C1',
        truth_color='black',
        quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        max_n_ticks=3
    )
    defaults_kwargs.update(kwargs)
    fig = corner.corner(xs, **defaults_kwargs)

    if fname is not None:
        fig.savefig(fname)

    return fig

def plot_multiple_bilby(xs_list, colors=None, fname=None, **kwargs):
    """ Make bilby-style corner plots with multiple data sets. """
    from matplotlib.lines import Line2D

    if colors is None:
        colors = ['#0072C1', '#FF8C00'] + [ f'C{i}' for i in range(2, len(xs_list)) ]

    xs_labels = kwargs.pop('xs_labels', None)
    default_kwargs = dict(color=colors[0])
    default_kwargs.update(kwargs)

    fig = plot_corner_bilby(xs_list[0], **default_kwargs)
    lines = []
    for i in range(1, len(xs_list)):
        default_kwargs['color'] = colors[i]
        plot_corner_bilby(xs_list[i], fname=None, fig=fig, **default_kwargs)

    if xs_labels is not None:
        for color in colors:
            lines.append(Line2D([0], [0], color=color, linestyle='-'))

    fig.legend(handles=lines, labels=xs_labels)

    if fname is not None:
        fig.savefig(fname)

    return fig
