import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use('classic')
font = {'fontname': 'Times New Roman', 'size': 14}
matplotlib.rcParams.update({'figure.figsize': [5., 5.], 'font.family': 'Times New Roman', 'figure.dpi': 350})
import collections

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_first_tick = 1

_iter = [_first_tick]
_xlabel = ['iterations']
_ticker_registry = collections.defaultdict(lambda: {})

_output_dir = ''
_stdout = True


def set_first_tick(first_tick):
    assert type(first_tick) is int and first_tick >= 0
    global _first_tick
    _first_tick = first_tick


def suppress_stdout():
    global _stdout
    _stdout = False


def set_output_dir(output_dir):
    global _output_dir
    _output_dir = output_dir


def set_xlabel_for_tick(index, label):
    global _xlabel
    if len(_xlabel) <= index:
        raise RuntimeWarning('plot_utils.py: xlabels doesn\'t have index %d' % (index))
    else:
        _xlabel[index] = label


def _enlarge_ticker(index):
    if len(_iter) > index:
        return
    _iter.extend([_first_tick] * (index + 1 - len(_iter)))
    _xlabel.extend(['iterations'] * (index + 1 - len(_xlabel)))


def tick(index=0):
    if len(_iter) <= index:
        _enlarge_ticker(index)
    _iter[index] += 1


def plot(name, value, index=0):
    if len(_iter) <= index:
        _enlarge_ticker(index)
    if name in _ticker_registry:
        if _ticker_registry[name] != index:
            raise ValueError('%s is not registered with the %d ticker!' % (name, index))
    else:
        _ticker_registry[name] = index
    if type(value) is tuple:
        _since_last_flush[name][_iter[index]] = np.array(value)
    else:
        _since_last_flush[name][_iter[index]] = value


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}: {}\t".format(name, np.mean(np.array(list(vals.values())), axis=0)))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = np.array([_since_beginning[name][x] for x in x_vals])
        index = _ticker_registry[name]
        plt.clf()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(y_vals.shape) == 1:
            plt.plot(x_vals, y_vals)
        else:  # with standard deviation
            plt.plot(x_vals, y_vals[:, 0])
            plt.fill_between(x_vals, y_vals[:, 0] - y_vals[:, 1], y_vals[:, 0] + y_vals[:, 1], alpha=0.5)
        plt.xlabel(_xlabel[index], **font)
        plt.ylabel(name, **font)
        ax.set_xlim(xmin=_first_tick)
        plt.savefig(os.path.join(_output_dir, name.replace(' ', '_') + '.png'), dpi=350)
        plt.close()

    if _stdout:
        print("iteration {}\t{}".format(_iter[index], "\t".join(prints)))
    _since_last_flush.clear()


def reset():
    global _since_beginning, _since_last_flush, _iter, _xlabel, _ticker_registry, _output_dir, _stdout
    _since_beginning = collections.defaultdict(lambda: {})
    _since_last_flush = collections.defaultdict(lambda: {})

    _iter = [_first_tick]
    _xlabel = ['iterations']
    _ticker_registry = collections.defaultdict(lambda: {})

    _output_dir = ''
    _stdout = True