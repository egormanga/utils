#!/usr/bin/python3
# Utils lib ProgressPool test

import time
from utils.nolog import log, Progress, ProgressPool, ThreadedProgressPool

pp = ProgressPool(
	Progress(2),
	Progress(3),
	Progress(2000),
)

for i in range(2):
	for j in range(3):
		for k in range(2000):
			pp.print(i, j, k)

pp.done()
time.sleep(1)

pp = ThreadedProgressPool(2)
pp.p[0].mv = 100
pp.start()

for i in range(50, 100):
	pp.cvs[0] = i+1
	pp.p[1].mv = i
	for j in range(i):
		pp.cvs[1] = j+1
		time.sleep(0.001)

pp.stop()
time.sleep(0.5)
del pp

log('Progress test ok\n')

# by Sdore, 2019
