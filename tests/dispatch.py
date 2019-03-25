#!/usr/bin/python3
# Utils lib dispatch() test

import typing
from utils.nolog import S, log, dispatch, dispatch_typecheck

def check(v, s):
	print(repr(v))
	assert v == s

def checkex(ex):
	print(f"{type(ex).__name__}: {ex}")
	assert ex

@dispatch
def f(x: int) -> str: return str(x)

@dispatch
def f(x: str) -> int: return int(x)

@dispatch
def f(x: str, y: int) -> str: return x+str(y)

@dispatch
def f(x, b: type(Ellipsis)) -> (int, str): return f".{x}."

@dispatch
def f(x, y: str, z=123) -> str: return str(x)+repr(y)+str(z)

@dispatch
def f(x: type(None)) -> int: return 'aoaoa'

try: print(f(...))
except TypeError as ex: e = ex
else: e = None
checkex(e)

try: print(f(None))
except TypeError as ex: e = ex
else: e = None
checkex(e)

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
checkex(e)

@dispatch
def h(x, y): return True

try: print(h(1))
except TypeError as ex: e = ex
else: e = None
checkex(e)

@dispatch
def k(*args): return args

check(k(1, 2), (1, 2))

@dispatch
def l(*args, **kwargs): return (args, kwargs)

check(l(1, 2, c='asd'), ((1, 2), {'c': 'asd'}))

@dispatch
def m(x: typing.List[int]): return str(x)

@dispatch
def m(x: typing.List[str]): return repr(x)

try: print(m([None]))
except TypeError as ex: e = ex
else: e = None
checkex(e)

check(m([1, 2, 3]), '[1, 2, 3]')
check(m(['a', 'b']), "['a', 'b']")

try: print(m([1, 2, '3']))
except TypeError as ex: e = ex
else: e = None
checkex(e)

class Test:
	def f(self, x):
		return S(' ').join((type(x), x))

class SubTest(Test):
	@dispatch
	def f(self, x: int):
		return S(' ').join(('This is an integer:', x, '!'))

	@dispatch
	def f(self, *args, **kwargs):
		return super().f(*args, **kwargs)

t = SubTest()
check(t.f('a'), "<class 'str'> a")
check(t.f(0), 'This is an integer: 0 !')

assert dispatch_typecheck(('  ', 7), typing.Tuple[str, int])

log('dispatch() test ok\n')

# by Sdore, 2019
