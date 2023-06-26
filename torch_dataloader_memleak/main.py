# ### Problem ###
# When a Python object is used with num_workers>0,
# each worker copies the object in its own process,
# leading to `num_workers+1`x memory usage.
# https://github.com/pytorch/pytorch/issues/13246
#
# ### Memory usage ###
# 1. Shared (RAM that is shared with other processes)
# 2. USS (RAM that is unique to its process)
# 3. PSS (USS + shared/num_processes)
# https://man7.org/linux/man-pages/man5/proc.5.html
#
# ### SEE ALSO ###
# https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
# https://pytorch-dev-podcast.simplecast.com/episodes/dataloader-with-multiple-workers-leaks-memory

from __future__ import annotations
from collections import defaultdict
import pickle
import sys
import torch
import json
from typing import Any
from tabulate import tabulate
import os
import time
import psutil

import torch
from torch.utils.data import DataLoader, Dataset


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res['rss'] += mmap.rss
        res['pss'] += mmap.pss
        res['uss'] += mmap.private_clean + mmap.private_dirty
        res['shared'] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith('/'):  # looks like a file path
            res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
    return res


class MemoryMonitor:
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
            self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)


class MyDataset(Dataset):
    def __init__(self):
        self.l = list(range(100_000_000))
        # self.l = torch.tensor(list(range(10_000_000)))

    def __getitem__(self, index):
        return self.l[index]

    def __len__(self):
        return len(self.l)


def main():
    ds = MyDataset()
    loader = DataLoader(
        ds,
        batch_size=10_000,
        num_workers=6,
        persistent_workers=True,  # True to collect memory usage at the end
    )
    iterator = iter(loader)

    monitor = MemoryMonitor()
    for w in iterator._workers:
        monitor.add_pid(w.pid)

    # higher shared
    # lower USS
    print(monitor.table())

    while True:
        try:
            next(iterator)
        except StopIteration:
            break

    # lower shared
    # higher USS
    print(monitor.table())


if __name__ == "__main__":
    main()
