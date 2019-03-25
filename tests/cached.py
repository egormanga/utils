#!/usr/bin/python3
# Utils lib cached*() test

from utils.nolog import log, cachedfunction, cachedproperty

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

log('cached*() test ok\n')

# by Sdore, 2019
