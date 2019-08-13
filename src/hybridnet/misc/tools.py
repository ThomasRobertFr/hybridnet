import signal

class DotDict(dict):
    def __init__(self, dic):
        super(DotDict, self).__init__(dic)
        for k, v in self.items():
            if type(v) == dict:
                self[k] = DotDict(v)

    def get_dict(self):
        out = {}
        for k, v in self.items():
            if type(v) == DotDict:
                out[k] = v.get_dict()
            else:
                out[k] = v
        return out

    def flatten(self):
        out = {}
        for k, v in self.items():
            if type(v) == DotDict:
                v = v.flatten()
                for k2, v2 in v.items():
                    out[k + "." + k2] = v2
            else:
                out[k] = v
        return out

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class _TimeoutExpired(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _TimeoutExpired


def input_with_timeout(prompt, timeout=30, default=""):
    stop_input = default
    try:
        # set signal handler
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds
        try:
            stop_input = input(prompt)
        finally:
            signal.alarm(0)  # cancel alarm
    except _TimeoutExpired:
        pass
    return stop_input

def image_grid(grid, plt, colorbar=False, line_titles=None):
    n = len(grid)
    p = len(grid[0])
    fig = plt.figure(figsize=(p*2, n*2))

    for i in range(n):
        for j in range(p):
            # display original
            ax = plt.subplot(n, p, i * p + j + 1)
            plt.imshow(grid[i][j].squeeze())
            if line_titles and j == 0:
                plt.ylabel(line_titles[i])
            if colorbar:
                plt.colorbar()
            plt.xticks([])
            plt.yticks([])
    return fig
