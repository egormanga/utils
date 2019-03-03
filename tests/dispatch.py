#!/usr/bin/python3
# Utils lib dispatch() test

from utils.nolog import dispatch

def check(v, s):
	print(repr(v))
	assert v == s

@dispatch
def f(x: int): return str(x)

@dispatch
def f(x: str): return int(x)

@dispatch
def f(x: str, y: int): return x+str(y)

@dispatch
def f(x, b: type(Ellipsis)): return f".{x}."

@dispatch
def f(x, y: str, z=123): return str(x)+repr(y)+str(z)

try: print(f(...))
except TypeError as ex: e = ex
else: e = None
print(repr(e))
assert e

@dispatch
def f(x): return 'lol'

check(f(1), '1')
check(f('1'), 1)
check(f('1', 2), '12')
check(f('1', y=2), '12')
check(f('asd', ...), '.asd.')
check(f('asd', b=...), '.asd.')
check(f('sqs', 'boo'), "sqs'boo'123")
check(f('sqs', 'boo', 219), "sqs'boo'219")
check(f('sqs', 'boo', z=0), "sqs'boo'0")
check(f(...), 'lol')

@dispatch
def g(a: (int, str)): return str(a)+type(a).__name__

check(g(1), '1int')
check(g('b'), 'bstr')
try: print(g(None))
except TypeError as ex: e = ex
else: e = None
print(repr(e))
assert e

@dispatch
def h(x, y): return True

try: print(h(1))
except TypeError as ex: e = ex
else: e = None
print(repr(e))
assert e

print('\ndispatch() test ok')

# by Sdore, 2019
