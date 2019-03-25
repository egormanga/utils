#!/usr/bin/python3
# Utils lib ProgressPool test

import time
from utils.nolog import log, ProgressPool, ThreadedProgressPool

pp = ProgressPool()
for i in pp.range(2+1):
	for j in pp.range(3+1):
		for k in pp.range(2000+1):
			pass
time.sleep(1)

pp = ThreadedProgressPool()
pp.start()
for i in pp.range(50, 100+1):
	for j in pp.range(i+1):
		time.sleep(0.001)
pp.stop()
time.sleep(0.5)

pp = ProgressPool()
for i in pp.iter((1, 2, 3, 4)):
	time.sleep(0.2)
del pp

log('Progress test ok\n')

# by Sdore, 2019
