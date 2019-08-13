"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np
import copy
import math

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


class Scheduler:
    def __init__(self, schedules, schedulables, nb_epochs=0):
        self.schedules = copy.deepcopy(schedules)
        self.schedulables = schedulables
        self.nb_epochs = nb_epochs
        self.init_val = {}

        for name in self.schedules:
            if name not in self.schedulables:
                raise ValueError("%s not schedulable" % name)
            if type(self.schedules[name]) != list:
                self.schedules[name] = [self.schedules[name]]
            for schedule in self.schedules[name]:
                self._preprocess_schedule(schedule)

        for name in self.schedulables:
            self.init_val[name] = self.schedulables[name]

    def __getitem__(self, key):
        return self.schedulables[key]

    def __getattr__(self, key):
        return self.schedulables[key]

    def _preprocess_schedule(self, schedule):
        start = schedule.get("start", 0)
        end = schedule.get("end", -1)

        # None: default
        if start is None:
            start = 0
        if end is None:
            end = -1

        # Check type
        if type(start) not in [int, float] or type(end) not in [int, float]:
            raise ValueError("Start and end must be numbers")

        # Check use of pos relative to end
        if (end < 0 or start < 0 or type(start) == float or type(end) == float) and self.nb_epochs == 0:
            raise ValueError("Need to set nb epochs to use relative epoch numbers")

        # Negative: position from end
        if start < 0:
            if type(start) == int: start = max(0, self.nb_epochs + start)
            else: raise ValueError("Negative start must be int")
        if end < 0 and type(end) == int:
            if type(end) == int: end = max(0, self.nb_epochs + end)
            else: raise ValueError("Negative end must be int")

        # Float: relative to training
        if type(start) == float:
            start = min(int((self.nb_epochs - 1) * start), self.nb_epochs)
        if type(end) == float:
            end = min(int((self.nb_epochs - 1) * end), self.nb_epochs)

        start = int(start)
        end = int(end)

        if "scheduler" not in schedule:
            t = str(schedule.get("type", ""))

            if not hasattr(self, "_schedule_"+t):
                raise ValueError("Scheduler %s unknown" % t)
            schedule["scheduler"] = getattr(self, "_schedule_" + t)

        schedule["start"] = start
        schedule["end"] = end

    def epoch(self, epoch):
        for name in self.schedules:

            val = self.init_val[name]
            for schedule in self.schedules[name]:
                val = self._apply_schedule(val, epoch, **schedule)

            self.schedulables[name] = val

    # Pre-scheduler
    def _apply_schedule(self, value, epoch, start, end, scheduler, **kwargs):
        if "deactivate_after" in kwargs and kwargs["deactivate_after"] >= epoch:
            return value
        if "deactivate_before" in kwargs and kwargs["deactivate_before"] <= epoch:
            return value
        t = (epoch - start) / (end - start) # t = 0..1 between start and end
        return scheduler(value, epoch=epoch, t=t, start=start, end=end, **kwargs)

    # Schedulers
    def _schedule_exp_up(self, value, t, K=5, **kwargs):
        return value * self._exp_ramp(1-t, K)

    def _schedule_exp_down(self, value, t, K=5, **kwargs):
        return value * self._exp_ramp(t, K)

    def _schedule_linear_up(self, value, t, value_start=0, value_end=None, **kwargs):
        return self._linear_thresholded(value_start, value if value_end is None else value_end, t)

    def _schedule_linear_down(self, value, t, value_start=None, value_end=0, **kwargs):
        return self._linear_thresholded(value if value_start is None else value_start, value_end, t)

    def _schedule_cosine_down(self, value, t, value_end=0, **kwargs):
        return value_end + (value - value_end) * self._cos_ramp(t)

    def _schedule_cosine_up(self, value, t, **kwargs):
        return value * self._cos_ramp(1-t)


    # Math functions
    def _exp_ramp(self, t, K):
        t = max(0, t)
        return math.exp(-K * (t ** 2))

    def _linear_thresholded(self, min_val, max_val, t):
        t = max(0, min(1, t))
        return min_val * (1-t) + max_val * t

    def _cos_ramp(current, t):
        t = max(0, min(1, t))
        return .5 * (math.cos(math.pi * t) + 1)



