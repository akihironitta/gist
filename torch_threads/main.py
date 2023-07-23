# https://github.com/pytorch/pytorch/issues/7087#issue-318787926
# $ for i in `seq 1 6`; do OMP_NUM_THREADS=$i python torch_threads/main.py; done
# the number of cpu threads: 1, time: 13.420310974121094
# the number of cpu threads: 2, time: 6.845336675643921
# the number of cpu threads: 3, time: 4.694216012954712
# the number of cpu threads: 4, time: 3.5995919704437256
# the number of cpu threads: 5, time: 2.9713845252990723
# the number of cpu threads: 6, time: 2.7180697917938232
import time

import numpy as np
import torch

INDEX = 100_000
a = torch.rand(INDEX, 1000)
index = np.random.randint(INDEX - 1, size=INDEX * 8)
b = torch.from_numpy(index)

start = time.time()
for _ in range(10):
    res = a.index_select(0, b)
print(
    "the number of cpu threads: {}, time: {}".format(
        torch.get_num_threads(), time.time() - start
    )
)
