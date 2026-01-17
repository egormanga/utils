#!/usr/bin/env python3
# Utils lib cached*() test

from utils.nolog import first, last, only

def check(v, s):
	print(repr(v))
	assert (v == s)

def checkex(ex):
        print(f"{type(ex).__name__}: {ex}")
        assert (ex)

l = list(range(10))

check(first(l), l[0])
check(last(l), l[-1])

check(first((), default=1), 1)
check(last((), default=2), 2)
check(only((), default=3), 3)

try: first('')
except StopIteration as ex: e = ex
else: e = None
checkex(e)

try: last('')
except StopIteration as ex: e = ex
else: e = None
checkex(e)

check(only('a'), 'a')

try: only('')
except StopIteration as ex: e = ex
else: e = None
checkex(e)

try: only('aa')
except StopIteration as ex: e = ex
else: e = None
checkex(e)

it = iter((1, 2, 3, 4))
first(it)
check(first(it), 2)
check(last(it), 4)

try: next(it)
except StopIteration as ex: e = ex
else: e = None
checkex(e)

# by Sdore, 2025
#  www.sdore.me
