#!/usr/bin/python3
# Utils lib cached*() test

import time
from utils.nolog import log, preeval, cachedfunction, cachedproperty

def check(v, s):
	print(repr(v))
	assert v == s

class Test:
	@property
	def nc(self):
		global called
		called = True
		print('nc() called')
		return 1234

	@cachedproperty
	def c(self):
		global called
		called = True
		print('c() called')
		return 2345

t = Test()

called = False
check(t.nc, 1234)
assert called

called = False
check(t.nc, 1234)
assert called

called = False
check(t.c, 2345)
assert called

called = False
check(t.c, 2345)
assert not called

@cachedfunction
def f(x):
	global called
	called = True
	print('f() called')
	return x**12

called = False
check(f(123), 123**12)
assert called

called = False
check(f(123), 123**12)
assert not called

called = False
check(f(1234), 1234**12)
assert called

def fnc(): return 1234**43210

@cachedfunction
def fcf(): return 1234**43210

@preeval
def fpe(): return 1234**43210

start = time.time()
for i in range(100): fnc()
print('f() no cache:', round(time.time()-start, 4), 'sec')

start = time.time()
for i in range(100): fcf()
print('f() cachedfunction():', round(time.time()-start, 4), 'sec')
assert time.time()-start < 0.5

start = time.time()
for i in range(100): fpe()
print('f() preeval():', round(time.time()-start, 4), 'sec')
assert time.time()-start < 0.001

log('cached*() test ok\n')

# by Sdore, 2019
