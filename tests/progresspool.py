#!/usr/bin/python3
# Utils lib ProgressPool test

import time
from utils.nolog import Progress, ProgressPool

pp = ProgressPool(
	Progress(5),
	Progress(10),
	Progress(20),
)

for i in range(5):
	for j in range(10):
		for k in range(20):
			pp.print(i+1, j+1, k+1)
			time.sleep(0.01)

# by Sdore, 2019
