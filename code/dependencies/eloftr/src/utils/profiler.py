from contextlib import contextmanager
import time

import torch


class PassThroughProfiler:
    @contextmanager
    def profile(self, action_name: str):
        yield action_name

    def start(self, action_name: str):
        return None

    def stop(self, action_name: str):
        return None

    def summary(self):
        return ""


class SimpleProfiler(PassThroughProfiler):
    def __init__(self):
        self._starts = {}
        self._durations = {}

    def start(self, action_name: str):
        self._starts[action_name] = time.perf_counter()

    def stop(self, action_name: str):
        start_time = self._starts.pop(action_name, None)
        if start_time is None:
            return
        self._durations.setdefault(action_name, []).append(time.perf_counter() - start_time)

    def summary(self):
        parts = []
        for action_name, durations in self._durations.items():
            total = sum(durations)
            parts.append(f"{action_name}: count={len(durations)}, total={total:.6f}s")
        return "\n".join(parts)

    @contextmanager
    def profile(self, action_name: str):
        self.start(action_name)
        try:
            yield action_name
        finally:
            self.stop(action_name)


class InferenceProfiler(SimpleProfiler):
    """
    This profiler records duration of actions with cuda.synchronize().
    """

    @contextmanager
    def profile(self, action_name: str):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start(action_name)
            yield action_name
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.stop(action_name)


class PyTorchProfiler(PassThroughProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__()


def build_profiler(name):
    if name == 'inference':
        return InferenceProfiler()
    if name == 'pytorch':
        return PyTorchProfiler()
    if name is None:
        return PassThroughProfiler()
    raise ValueError(f'Invalid profiler: {name}')
