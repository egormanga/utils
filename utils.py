#!/usr/bin/python3
# Utils lib

""" Sdore's Utils

Code example:

from utils import *; logstart('<NAME>')

def main(): pass

if (__name__ == '__main__'): logstarted(); main()
else: logimported()

"""

import inspect
from pprint import pprint, pformat

module = type(inspect)

class ModuleProxy(module):
	__slots__ = ('_as',)

	def __init__(self, name, as_=None):
		self.__name__, self._as = name, as_ or name

	def __getattribute__(self, x):
		module = __import__(object.__getattribute__(self, '__name__'))
		inspect.stack()[1][0].f_globals[object.__getattribute__(self, '_as')] = module
		return getattr(module, x)

def Simport(x, as_=None):
	#inspect.stack()[1][0].f_globals
	globals()[as_ or x] = ModuleProxy(x, as_)

_imports = (
	'io',
	'os',
	're',
	'abc',
	'ast',
	'bs4',
	'cmd',
	'dis',
	'pip',
	'pty',
	'sys',
	'tty',
	'code',
	'copy',
	'dill',
	'glob',
	'html',
	'json',
	'math',
	'stat',
	'time',
	'uuid',
	'zlib',
	'queue',
	'regex',
	'shlex',
	'base64',
	'bidict',
	'codeop',
	'getkey',
	'locale',
	'parser',
	'pickle',
	'psutil',
	'random',
	'select',
	'shutil',
	'signal',
	'socket',
	'string',
	'struct',
	'typing',
	'asyncio',
	'aiohttp',
	'bashlex',
	'getpass',
	'hashlib',
	'keyword',
	'marshal',
	'numbers',
	'os.path',
	'termios',
	'zipfile',
	'argparse',
	'attrdict',
	'builtins',
	'datetime',
	'operator',
	'platform',
	'readline',
	'requests',
	'tempfile',
	'fractions',
	'functools',
	'importlib',
	'itertools',
	'pyparsing',
	'threading',
	'traceback',
	'contextlib',
	'subprocess',
	'collections',
	'rlcompleter',
	'better_exchook',
	'typing_inspect',
	'multiprocessing_on_dill as multiprocessing',
	#'nonexistenttest'
)
for i in _imports: Simport(*i.split()[::2])
del i, Simport # TODO FIXME? (inspect.stack() is too slow)

def install_all_imports():
	r = list()
	for i in _imports:
		i = i.partition(' ')[0]
		if (i in sys.modules): continue
		try: __import__(i)
		except ModuleNotFoundError: r.append(i)
	old_sysargv, sys.argv = sys.argv, ['pip3', 'install']+r
	try: __import__('pkg_resources').load_entry_point('pip', 'console_scripts', 'pip3')()
	finally: sys.argv = old_sysargv

py_version = 'Python '+sys.version.split(maxsplit=1)[0]

#try: better_exchook.install()
#except ModuleNotFoundError: pass

argparser = argparse.ArgumentParser(conflict_handler='resolve', add_help=False)
argparser.add_argument('-v', action='count', help=argparse.SUPPRESS)
argparser.add_argument('-q', action='count', help=argparse.SUPPRESS)
cargs = argparser.parse_known_args()[0]
argparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
loglevel = (cargs.v or 0)-(cargs.q or 0)

function = type(lambda: None)
generator = type(_ for _ in ())
CodeType = type(compile('', '', 'exec'))
NoneType = type(None)
inf = float('inf')
nan = float('nan')
endl = '\n'

def isiterable(x): return isinstance(x, typing.Iterable)
def isiterablenostr(x): return isiterable(x) and not isinstance(x, str)
def isnumber(x): return isinstance(x, (int, float, complex))
def parseargs(kwargs, **args): args.update(kwargs); kwargs.update(args); return kwargs
def hex(x, l=2): return '0x%%0%dX' % l % x
def randstr(n=16): return str().join(random.choices(string.ascii_letters, k=n))

def S(x=None):
	""" Convert `x' to an instance of corresponding S-type. """
	ex = None
	try: return eval('S'+type(x).__name__)(x) if (not isinstance(x, Stype)) else x
	except NameError: ex = True
	if (ex): raise NotImplementedError("S%s" % type(x).__name__)

class Stype: pass

class Sdict(Stype, collections.defaultdict):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if (not args or isiterable(args[0])): args.insert(0, None)
		super().__init__(*args, **kwargs)

	__repr__ = dict.__repr__

	def __getattr__(self, x):
		try: return self[x]
		except KeyError: pass
		raise AttributeError(x)

	def __setattr__(self, x, v):
		try: self[x] = v
		except KeyError: pass
		else: return
		raise AttributeError(x)

	def __and__(self, x):
		""" Return self with applied .update(x). """
		r = Sdict(self); r.update(x); return r

	def __matmul__(self, x):
		""" Return list of values in self filtered by `x'. """
		if (isinstance(x, dict)): return Slist(i for i in self if all(self[i] in j for j in x.values())) # TODO doc me
		#if (len(x) == 1): return Slist(self[i].get(x[0]) for i in self) # TODO wtf??
		return Slist(self.get(i) for i in x)

	def __call__(self, *keys):
		""" Return subdict of self filtered by call arguments (`*keys'). """
		return Sdict({i: self.get(i) for i in keys})

	def __copy__(self):
		return Sdict(self.default_factory, self)

	copy = __copy__

	def translate(self, table, copy=False, strict=True, keep=True):
		r = Sdict(self)
		for i in table:
			k, t = table[i] if (isinstance(table[i], (tuple, list))) else (table[i], lambda x: x)
			if (not strict and k not in r): continue
			if (keep and i not in r): r[i] = t((r.get if (copy) else r.pop)(k))
		return r

	def with_(self, key, value=None):
		""" Return self with item `key' set to `value'. """
		r = self.copy()
		r[key] = value
		return r

	_to_discard = set()

	def to_discard(self, x):
		self._to_discard.add(x)

	def discard(self):
		for i in self._to_discard:
			try: self.pop(i)
			except IndexError: pass
		self._to_discard.clear()
Sdefaultdict = Sdict

class Slist(Stype, list):
	def __matmul__(self, item):
		if (type(item) == dict): return Slist(i for i in self if all(i.get(j) in (item[j]) for j in item))
		r = Slist(Slist((i.get(j) if (hasattr(i, 'get')) else i[j]) for j in item) for i in self)
		return r.flatten() if (len(item) == 1 and not isiterable(item[0]) or type(item[0]) == str) else r

	def __getitem__(self, x):
		if (not isiterable(x)): return list.__getitem__(self, x)
		return Slist(i for i in self if (type(i) == dict and i.get(x[0]) in x[1:]))

	def __sub__(self, x):
		l = self.copy()
		for i in range(len(l)):
			if (all(l[i][j] == x[j] for j in x)): l.to_discard(i)
		l.discard()
		return l

	def copy(self):
		return Slist(self)

	def rindex(self, x, start=0): # TODO FIXME
		l = len(self)
		return l-self.index(x, 0, l-start)-1

	def group(self, n):
		return Slist((*(j for j in i if (j is not None)),) for i in itertools.zip_longest(*[iter(self)]*n))

	def flatten(self):
		return Slist(j for i in self if i for j in (i if (i and isiterable(i) and not isinstance(i, str)) else (i,)))

	def strip(self, s=None):
		l = self.copy()
		if (not isiterable(s) or isinstance(s, str)): s = (s,)
		return Slist(i for i in l if (i if (s is None) else i not in s))

	def filter(self, t):
		return Slist(i for i in self if type(i) == t)

	def uniquize(self, key=lambda x: x):
		was = set()
		return Slist(was.add(key(i)) or i for i in self if key(i) not in was) # such a dirty hack..

	#def wrap(self, 

	_to_discard = set()
	def to_discard(self, x):
		self._to_discard.add(x)

	def discard(self):
		for i in self._to_discard:
			try: self.pop(i)
			except IndexError: pass
		self._to_discard.clear()

class Stuple(Slist): pass # TODO

class Sint(Stype, int):
	def __len__(self):
		return Sint(math.log10(abs(self) or 10))

	def constrain(self, lb, ub):
		return Sint(constrain(self, lb, ub))

	def format(self, char=' '):
		return char.join(map(str().join, Slist(str(self)[::-1]).group(3)))[::-1]

	def pm(self):
		return f"+{self}" if (self > 0) else str(self)

class Sstr(Stype, str):
	_subtrans = str.maketrans({
		'0': '₀',
		'1': '₁',
		'2': '₂',
		'3': '₃',
		'4': '₄',
		'5': '₅',
		'6': '₆',
		'7': '₇',
		'8': '₈',
		'9': '₉',
		'+': '₊',
		'-': '₋',
		'=': '₌',
		'(': '₍',
		')': '₎',
		'a': 'ₐ',
		'e': 'ₑ',
		'o': 'ₒ',
		'x': 'ₓ',
		'ə': 'ₔ',
		'h': 'ₕ',
		'k': 'ₖ',
		'l': 'ₗ',
		'm': 'ₘ',
		'n': 'ₙ',
		'p': 'ₚ',
		's': 'ₛ',
		't': 'ₜ',
	})

	_suptrans = str.maketrans({
		'0': '⁰',
		'1': '¹',
		'2': '²',
		'3': '³',
		'4': '⁴',
		'5': '⁵',
		'6': '⁶',
		'7': '⁷',
		'8': '⁸',
		'9': '⁹',
		'.': '·',
		'+': '⁺',
		'-': '⁻',
		'=': '⁼',
		'(': '⁽',
		')': '⁾',
		'i': 'ⁱ',
		'n': 'ⁿ',
	})

	def __getitem__(self, x):
		return Sstr(str.__getitem__(self, x))

	def __and__(self, x):
		return Sstr().join(i for i in self if i in x)

	def fit(self, l, *, end='…'):
		return Sstr(self if (len(self) <= l) else self[:l-1]+end)

	def cyclefit(self, l, n, *, sep=' '*8, start_delay=0, **kwargs):
		if (len(self) <= l): return self
		n = max(0, (n % (len(self)+len(sep)+start_delay))-start_delay)
		return Sstr((self+sep)[n:]+self[:n]).fit(l)

	def join(self, l, *, first='', last=None):
		l = tuple(map(str, l))
		r = (str.join(self, l[:-1])+(last or self)+l[-1]) if (len(l) > 1) else l[0] if (l) else ''
		if (r): r = first+r
		return Sstr(r)

	def bool(self, minus_one=True):
		return bool(self) and self.casefold() not in ('0', 'false', 'no', 'нет', '-1'*(not minus_one))

	def indent(self, n=None, char=None, tab_width=8):
		if (not self): return self
		if (n is None): n = tab_width
		r, n = n < 0, abs(n)
		if (char is None): char = ('\t'*(n//tab_width)+' '*(n % tab_width)) if (not r) else ' '*n
		else: char *= n
		return Sstr(char+('\n'+char).join(self.split('\n')) if (not r) else (char+'\n').join(self.split('\n'))+char)

	def just(self, n, char=' ', j=None):
		if (j is None): j, n = '<>'[n>0], abs(n)
		if (j == '.'): r = self.center(n, char)
		elif (j == '<'): r = self.ljust(n, char)
		elif (j == '>'): r = self.rjust(n, char)
		else: raise ValueError(j)
		return Sstr(r)

	def sjust(self, n, *args, **kwargs):
		return self.indent(n-len(self), *args, **kwargs)

	def wrap(self, w, j='<', char=' ', loff=0, sep=' '):
		if (len(self) <= w-loff): return self
		r = Sstr()
		s = self.split(sep)
		for i in s[:-1]:
			r += i
			r += '\n' if (len(r.split('\n')[-1])+len(i)+len(sep) > w-loff) else sep
		r += s[-1]
		return Sstr('\n'.join(Sstr(char*(loff*bool(ii))+i) for ii, i in enumerate(r.rstrip(' ').split('\n'))))

	def split(self, *args, **kwargs):
		return Slist(map(Sstr, str.split(self, *args, **kwargs)))

	def filter(self, chars):
		return Sstr().join(i for i in self if i in chars)

	def capwords(self, sep=' '):
		s = list(self)
		c = True
		for ii, i in enumerate(s):
			if (c): s[ii] = i.upper(); c = False
			if (i in sep): c = True
		return Sstr().join(s)

	def sub(self):
		return self.translate(self._subtrans)

	def super(self): logexception(DeprecationWarning(" *** super() → sup() *** ")); return self.sup()
	def sup(self):
		return self.translate(self._suptrans)

def Sbool(x=bool(), *args, **kwargs): # No way to derive a class from bool
	x = S(x)
	return x.bool(*args, **kwargs) if (hasattr(x, 'bool')) else bool(x)

def code_with(code, **kwargs):
	if (sys.version_info >= (3, 8)): return code.replace(**{'co_'+k: v for k, v in kwargs.items()})
	return CodeType(*(kwargs.get(i, getattr(code, 'co_'+i)) for i in ('argcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', 'lnotab', 'freevars', 'cellvars')))
# TODO: func_with()

class DispatchFillValue: pass
class DispatchError(TypeError): pass
def dispatch_typecheck(o, t):
	if (t is None or t is inspect._empty): return True
	if (isinstance(o, DispatchFillValue) or isinstance(t, DispatchFillValue)): return False
	if (isinstance(t, function)): return bool(t(o))
	if (not isinstance(o, (typing_inspect.get_constraints(t) or type(o)) if (isinstance(t, typing.TypeVar)) else typing_inspect.get_origin(t) or t)): return False
	if (isinstance(o, typing.Tuple) and typing_inspect.get_origin(t) and issubclass(typing_inspect.get_origin(t), typing.Tuple) and typing_inspect.get_args(t) and not all(itertools.starmap(dispatch_typecheck, itertools.zip_longest(o, typing_inspect.get_args(t), fillvalue=DispatchFillValue())))): return False
	if (isinstance(o, typing.Iterable) and not isinstance(o, (typing.Iterator)) and typing_inspect.get_args(t) and not all(dispatch_typecheck(i, typing_inspect.get_args(t)[0]) for i in o)): return False
	return True
_overloaded_functions = Sdict(dict)
_overloaded_functions_retval = Sdict(dict)
_overloaded_functions_docstings = Sdict(dict)
def dispatch(f):
	fname = f.__qualname__
	if (getattr(f, '__signature__', None) == ...): del f.__signature__ # TODO FIXME ???
	fsig = inspect.signature(f)
	params_annotation = tuple((i[0], (None if (i[1].annotation is inspect._empty) else i[1].annotation, i[1].default is not inspect._empty, i[1].kind)) for i in fsig.parameters.items())
	_overloaded_functions[fname][params_annotation] = f
	_overloaded_functions_retval[fname][params_annotation] = type(fsig.return_annotation) if (fsig.return_annotation is None) else fsig.return_annotation
	_overloaded_functions_docstings[fname][fsig] = f.__doc__
	#dplog([(dict(i), '—'*40) for i in _overloaded_functions[fname]], width=60) # XXX
	def overloaded(*args, **kwargs):
		args = list(args)
		for k, v in _overloaded_functions[fname].items():
			#dplog(k) # XXX
			i = int()
			no = True
			for ii, a in enumerate(args):
				if (i >= len(k)): break
				if (not dispatch_typecheck(a, k[i][1][0])): break
				if (k[i][1][2] in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)): i += 1
			else: no = False
			if (no): continue

			if (i < len(k) and k[i][1][2] == inspect.Parameter.VAR_POSITIONAL): i += 1

			no = True
			kw = dict(k[i:])
			pkw = set()
			varkw = tuple(filter(lambda x: x[2] == inspect.Parameter.VAR_KEYWORD, kw.values()))
			for a in kwargs:
				if (a not in kw):
					if (not varkw): break
					ckw = varkw[0]
				else: ckw = kw[a]
				if (not dispatch_typecheck(kwargs[a], ckw[0])): break
				pkw.add(a)
			else: no = False
			if (no): continue
			if (not varkw and {i for i in kw if (not kw[i][1])}-pkw): continue

			r = \
				v(*args, **kwargs)
			break
		else:
			if (() in _overloaded_functions[fname]): r = _overloaded_functions[fname][()](*args, **kwargs)
			else:
				ex = DispatchError(f"Parameters {S(', ').join((*map(type, args), *map(type, kwargs.values()))).join('()')} don't match any of '{fname}' signatures:\n{S(overloaded_format_signatures(fname, f.__qualname__, sep=endl)).indent(2, char=' ')}") # to hide in tb
				raise ex
		retval = _overloaded_functions_retval[fname][k]
		if (retval is not inspect._empty and not isinstance(r, retval)):
			raise DispatchError(f"Return value of type {type(r)} doesn't match return annotation of appropriate '{fname}' signature")
		return r
	overloaded.__name__, overloaded.__qualname__, overloaded.__module__, overloaded.__doc__, overloaded.__signature__, overloaded.__code__ = f"Overloaded {f.__name__}", f.__qualname__, f.__module__, (_overloaded_functions_docstings[fname][()]+'\n\n' if (() in _overloaded_functions_docstings[fname]) else '')+overloaded_format_signatures(fname, f.__qualname__), ..., code_with(overloaded.__code__, name=f"<overload handler of '{f.__qualname__}'>")
	f.__name__, f.__code__ = f"Overloaded {f.__name__}", code_with(f.__code__, name=f"<overloaded '{f.__qualname__}' for {f.__name__}{format_inspect_signature(fsig)}>")
	return overloaded
def dispatch_meta(f):
	if (f.__doc__): _overloaded_functions_docstings[f.__qualname__][()] = f.__doc__
	return f
def overloaded_format_signatures(fname, qualname, sep='\n\n'): return sep.join(Sstr().join((qualname, format_inspect_signature(fsig), ':\n\b    '+doc if (doc) else '')) for fsig, doc in _overloaded_functions_docstings[fname].items())

def format_inspect_signature(fsig):
	result = list()
	posonlysep = False
	kwonlysep = True

	for p in fsig.parameters.values():
		if (p.kind == inspect.Parameter.POSITIONAL_ONLY): posonlysep = True
		elif (posonlysep): result.append('/'); posonlysep = False
		if (p.kind == inspect.Parameter.VAR_POSITIONAL): kwonlysep = False
		elif (p.kind == inspect.Parameter.KEYWORD_ONLY and kwonlysep): result.append('*'); kwonlysep = False
		result.append(f"{'*' if (p.kind == inspect.Parameter.VAR_POSITIONAL) else '**' if (p.kind == inspect.Parameter.VAR_KEYWORD) else ''}{p.name}{f': {inspect.formatannotation(p.annotation)}' if (p.annotation is not inspect._empty) else ''}{f' = {p.default}' if (p.annotation is not inspect._empty and p.default is not inspect._empty) else f'={p.default}' if (p.default is not inspect._empty) else ''}")

	if (posonlysep): result.append('/')
	rendered = ', '.join(result).join('()')
	if (fsig.return_annotation is not inspect._empty): rendered += f" -> {inspect.formatannotation(fsig.return_annotation)}"
	return rendered

def cast(*types): return lambda x: (t(i) if (not isinstance(i, t)) else i for t, i in zip(types, x))

def cast_call(f, *args, **kwargs):
	fsig = inspect.signature(f)
	r = f(*((v.annotation)(args[ii]) if (v.annotation is not inspect._empty and not isinstance(args[ii], v.annotation)) else args[ii] for ii, (k, v) in enumerate(fsig.parameters.items()) if ii < len(args)), **{k: (fsig.parameters[k].annotation)(v) if (k in fsig.parameters and fsig.parameters[k].annotation is not inspect._empty and not isinstance(v, fsig.parameters[k].annotation)) else v for k, v in kwargs.items()})
	return (fsig.return_annotation)(r) if (fsig.return_annotation is not inspect._empty) else r

def autocast(f):
	""" Beware! leads to undefined behavior when used with @dispatch. """
	r = lambda *args, **kwargs: cast_call(f, *args, **kwargs)
	r.__annotations__ = f.__annotations__
	return r

def init_defaults(f):
	""" Decorator that initializes type-annotated arguments which are not specified on function call.
	You can also use lambdas as annotations.
	Useful for initializing mutable defaults.
	Note: you should not specify default values for these arguments or it will override the initialization.

	Example:
		@init_defaults
		def f(x: list):
			x.append(1)
			return x
		f() #--> [1]
		a = f() #--> [1]
		f(x=a) #--> [1, 1]

	Has no bugs.
	"""
	return lambda *args, **kwargs: f(*args, **S(kwargs) & {k: v() for k, v in f.__annotations__.items() if k not in kwargs})
	#fsig = inspect.signature(f)
	#return lambda *args, **kwargs: f(*args, **S(kwargs) & {k: v.annotation() for k, v in fsig.parameters.items() if (k not in kwargs and v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and v.annotation is not inspect._empty)})

class cachedfunction: # TODO FIXME DEPRECATION (--> @functools.lru_cache; ~100 times faster)
	def __init__(self, f):
		self.f = f
		self._cached = dict()
		self._obj = None

	def __get__(self, obj, objcls):
		self._obj = obj
		return self

	def __call__(self, *args, **kwargs):
		k = tohashable((args, kwargs))
		if (self._obj is not None): args = (self._obj, *args)
		if (k not in self._cached): self._cached[k] = self.f(*args, **kwargs)
		return self._cached[k]

	def nocache(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		return self.f(*args, **kwargs)

	def clear_cache(self):
		self._cached.clear()
class cachedclass(cachedfunction): pass

class cachedproperty: # TODO FIXME DEPRECATION (--> @functools.cached_property; ~100 times faster)
	class _empty: __slots__ = ()
	_empty = _empty()

	def __init__(self, f):
		self.f = f
		self._cached = self._empty

	def __get__(self, obj, objcls):
		if (self._cached is self._empty): self._cached = self.f(obj)
		return self._cached

	def clear_cache(self):
		self._cached.clear()

@dispatch
def allsubclasses(cls: type):
	""" Get all subclasses of class `cls' and its subclasses and so on recursively. """
	return cls.__subclasses__()+[j for i in cls.__subclasses__() for j in allsubclasses(i)]
@dispatch
def allsubclasses(obj): return allsubclasses(type(obj))

@dispatch
def subclassdict(cls: type):
	""" Get name-class mapping for subclasses of class `cls'. """
	return {i.__name__: i for i in cls.__subclasses__()}
@dispatch
def subclassdict(obj): return subclassdict(type(obj))

def funcdecorator(df):
	def ndf(f, *args, **kwargs):
		if (not isinstance(f, function)): return lambda nf: ndf(nf, f, *args, **kwargs)
		nf = df(f)
		nf.__name__, nf.__qualname__, nf.__module__, nf.__doc__, nf.__signature__ = \
		 f.__name__,  f.__qualname__,  f.__module__,  f.__doc__, inspect.signature(f)
		nf.__code__ = code_with(nf.__code__, name=f"<decorated '{f.__code__.co_name}'>" if (not f.__code__.co_name.startswith('<decorated ')) else f.__code__.co_name)
		for i in filter(lambda x: not x.startswith('__'), dir(f)):
			setattr(nf, i, getattr(f, i))
		return nf

	#dfsig, ndfsig = inspect.signature(df), inspect.signature(ndf)
	#if (dfsig != ndfsig): raise ValueError(f"Function decorated with @funcdecorator should have signature '{format_inspect_signature(ndfsig)}' (got: '{format_inspect_signature(dfsig)}')") # TODO kwargs

	ndf.__name__, ndf.__qualname__, ndf.__module__, ndf.__doc__, ndf.__code__ = \
	 df.__name__,  df.__qualname__,  df.__module__,  df.__doc__, code_with(ndf.__code__, name=df.__code__.co_name)
	for i in filter(lambda x: not x.startswith('__'), dir(df)):
		setattr(nff, i, getattr(df, i))

	return ndf

def spreadargs(f, okwargs, *args, **akwargs):
	fsig = inspect.signature(f)
	kwargs = S(okwargs) & akwargs
	try: kwnames = (*(i[0] for i in fsig.parameters.items() if (assert_(i[1].kind != inspect.Parameter.VAR_KEYWORD) and i[1].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))),)
	except AssertionError: kwnames = kwargs.keys()
	for i in kwnames:
		try: del okwargs[i]
		except KeyError: pass
	return f(*args, **kwargs)

def init(*names):
	@funcdecorator
	def decorator(f):
		def decorated(self, *args, **kwargs):
			for i in names:
				setattr(self, i, kwargs.pop(i))
			return f(self, *args, **kwargs)
		return decorated
	return decorator

logcolor = ('\033[94m', '\033[92m', '\033[93m', '\033[91m', '\033[95m')
noesc = re.compile(r'\033\[?[0-?]*[ -/]*[@-~]?')
logfile = None
logoutput = sys.stderr
loglock = queue.LifoQueue()
_logged_utils_start = None
def log(l=None, *x, sep=' ', end='\n', ll=None, raw=False, tm=None, format=False, width=80, unlock=False, nolog=False): # TODO: finally rewrite me as class pls
	""" Log anything. Print (formatted with datetime) to stderr and logfile (if set). Should be compatible with builtins.print().
	Parameters:
		[l]: level, must be >= global loglevel to print to stderr rather than only to logfile (or /dev/null).
		*x: print-like args, what to log.
		sep: print-like separator.
		end: print-like line end.
		ll: formatted string instead of loglevel for output.
		raw: if true, do not print datetime/loglevel prefix if set.
		tm: specify datetime, time.time() if not set, nothing if false.
		format: if true, apply pformat() to args.
		width: specify output line width for wrapping, autodetect from stderr if not specified.
		unlock: if true, release all previously holded («locked») log messages.
		nolog: if true, force suppress printing to stderr.
	"""
	global loglock, _logged_utils_start
	if (isinstance(_logged_utils_start, tuple)): _logged_utils_start, _logstateargs = True, _logged_utils_start; logstart('Utils'); logstate(*_logstateargs)
	_l, _x = l, x#map(copy.copy, (l, x))
	if (l is None): l = ''
	if (type(l) is not int): l, x = None, (l, *x)
	if (x == ()): l, x = 0, (l,)
	x = 'plog():\n'*bool(format and not raw)+sep.join(map((lambda x: pformat(x, width=width)) if (format) else str, x))
	clearstr = noesc.sub('', str(x))
	if (tm is None): tm = time.localtime()
	if (not unlock and not loglock.empty()): loglock.put(((_l, *_x), dict(sep=sep, end=end, raw=raw, tm=tm, nolog=nolog))); return clearstr
	try: lc = logcolor[l]
	except (TypeError, IndexError): lc = ''
	if (ll is None): ll = f'[\033[1m{lc}LV{l}\033[0;96m]' if (l is not None) else ''
	logstr = f"\033[K\033[96m{time.strftime('[%x %X] ', tm) if (tm) else ''}\033[0;96m{ll}\033[0;1m{' '*bool(ll)+x if (x != '') else ''}\033[0m{end}" if (not raw) else str(x)+end
	if (unlock and not loglock.empty()):
		ul = list()
		for i in iter_queue(loglock):
			if (i is None): break
			ul.append(i)
		for i in ul[::-1]:
			log(*i[0], **i[1])
	if (logfile and not nolog): logfile.write(logstr)
	if ((l or 0) <= loglevel): logoutput.write(logstr); logoutput.flush()
	return clearstr
def plog(*args, **kwargs): parseargs(kwargs, format=True); return log(*args, **kwargs)
def dlog(*args, **kwargs): parseargs(kwargs, ll='\033[95m[\033[1mDEBUG\033[0;95m]\033[0;96m', tm=''); return log(*((args[0] if (args[0] is not None) else repr(args[0]),) if (args) else ()), *args[1:], **kwargs)
def dplog(*args, **kwargs): parseargs(kwargs, format=True, sep='\n'); return dlog(*args, **kwargs)
def rlog(*args, **kwargs): parseargs(kwargs, raw=True); return log(*args, **kwargs)
def logdumb(**kwargs): return log(raw=True, end='', **kwargs)
def logstart(x):
	""" from utils import *; logstart(name) """
	global _logged_utils_start
	if (_logged_utils_start is None): _logged_utils_start = False; return
	log(x+'\033[0m...', end=' ', nolog=(x == 'Utils'))
	locklog()
def logstate(state, x=''):
	global _logged_utils_start
	if (_logged_utils_start is False): _logged_utils_start = (state, x); return
	log(state+(': '+str(x))*bool(str(x))+'\033[0m', raw=True, unlock=True)
def logstarted(x=''): """ if (__name__ == '__main__'): logstart(); main() """; logstate('\033[94mstarted', x)
def logimported(x=''): """ if (__name__ != '__main__'): logimported() """; logstate('\033[96mimported', x)
def logok(x=''): logstate('\033[92mok', x)
def logex(x=''): logstate('\033[91merror', unraise(x))
def logwarn(x=''): logstate('\033[93mwarning', x)
def setlogoutput(f): global logoutput; logoutput = f
def setlogfile(f): global logfile; logfile = open(f, 'a')
def setloglevel(l): global loglevel; loglevel = l
def locklog(): loglock.put(None)
def unlocklog(): logdumb(unlock=True)
def setutilsnologimport(): global _logged_utils_start; _logged_utils_start = True

_exc_handlers = set()
def register_exc_handler(f): _exc_handlers.add(f)
_logged_exceptions = set()
@dispatch
def exception(ex: BaseException, once=False, raw=False, nolog=False):
	""" Log an exception.
	Parameters:
		ex: exception to log.
		nolog: no not call exc_handlers.
	"""
	ex = unraise(ex)
	if (isinstance(ex, NonLoggingException)): return
	if (once):
		if (repr(ex) in _logged_exceptions): return
		_logged_exceptions.add(repr(ex))
	exc = repr(ex).partition('(')[0]
	e = log(('\033[91mCaught ' if (not isinstance(ex, Warning)) else '\033[93m' if ('warning' in ex.__class__.__name__.casefold()) else '\033[91m')+f"{exc}{(' on line '+'→'.join(map(lambda x: str(x[1]), traceback.walk_tb(ex.__traceback__)))).rstrip(' on line')}\033[0m{(': '+str(ex))*bool(str(ex))}", raw=raw)
	if (nolog): return
	for i in _exc_handlers:
		try: i(e, ex)
		except Exception: pass
logexception = exception

def raise_(ex): raise ex

@dispatch
def unraise(ex: BaseException): return ex
@dispatch
def unraise(ex: type):
	if (not issubclass(ex, BaseException)): raise TypeError()
	return ex()

@dispatch
def catch(*ex: BaseException):
	@funcdecorator
	def decorator(f):
		def decorated(*args, **kwargs):
			with contextlib.suppress(*ex):
				return f(*args, **kwargs)
		return decorated
	return decorator
@dispatch
def catch(f: function): return catch(BaseException)(f)

@dispatch
def suppress(*ex: BaseException):
	raise TODO

def assert_(x): assert x; return True

def noop(*_, **__): pass

class _clear:
	""" Clear the terminal. """

	def __call__(self):
		print(end='\033c', flush=True)

	def __repr__(self):
		self(); return ''
clear = _clear()

class DB:
	""" All-in-one lightweight database class. """

	def __init__(self, file=None, serializer=pickle):
		self.setfile(file)
		self.setserializer(serializer)
		self.fields = dict()
		self.setnolog(False)
		self.setbackup(True)

	def setfile(self, file):
		self.file = file
		if (file is None): return False
		#if ('/' not in file): file = os.path.dirname(os.path.realpath(sys.argv[0]))+'/'+file
		try: self.file = open(file, 'r+b')
		except FileNotFoundError: self.file = open(file, 'w+b')
		return True

	def setnolog(self, nolog=True):
		self.nolog = bool(nolog)

	def setbackup(self, backup):
		self.backup = bool(backup)

	@dispatch
	def setserializer(self, serializer: module):
		self.serializer = serializer

	def register(self, *fields):
		globals = inspect.stack()[1][0].f_globals
		for field in fields: self.fields[field] = globals

	def load(self, nolog=None):
		nolog = (self.nolog if (nolog is None) else nolog)
		if (not self.file): return
		if (not nolog): logstart('Loading database')
		db = dict()
		try: db = self.serializer.load(self.file);
		except EOFError:
			if (not nolog): logwarn('database is empty')
		else:
			if (not nolog): logok()
		self.file.seek(0)
		for field in self.fields:
			if (field in db): self.fields[field][field] = db.get(field)
			elif (not nolog): log(1, f"Not in DB: {field}")
		return db

	def save(self, db={}, backup=None, nolog=None):
		nolog = (self.nolog if (nolog is None) else nolog)
		backup = (self.backup if (backup is None) else backup)
		if (not self.file): return
		if (not nolog): logstart('Saving database')
		if (backup):
			try: os.mkdir('backup')
			except FileExistsError: pass
			try: shutil.copyfile(self.file.name, f"backup/{self.file.name if (hasattr(self.file, 'name')) else ''}_{int(time.time())}.db")
			except OSError: pass
		try: self.serializer.dump(db or {field: self.fields[field][field] for field in self.fields}, self.file)
		except Exception as ex:
			if (not nolog): logex(ex)
		else:
			if (not nolog): logok()
		self.file.truncate()
		self.file.seek(0)
db = DB()

def progress(cv, mv, pv="▏▎▍▌▋▊▉█", fill='░', border='│', prefix='', print=True): # TODO: maybe deprecate 🤔
	return getattr(Progress(mv, chars=pv, border=border, prefix=prefix), 'print' if (print) else 'format')(cv)

class Progress:
	def __init__(self, mv=-1, chars=' ▏▎▍▌▋▊▉█', border='│', prefix='', add_speed_eta=True):
		self.mv, self.chars, self.border, self.prefix, self.add_speed_eta = mv, chars, border, prefix, add_speed_eta
		self.fstr = self.prefix+'%d/%d (%d%%%s) '
		self.printed = bool()
		self.started = None

	def __del__(self):
		if (self.printed): sys.stderr.write('\n')

	def format(self, cv, width, add_speed_eta=None):
		if (add_speed_eta is None): add_speed_eta = self.add_speed_eta
		if (self.started is None): self.started = time.time(); add_speed_eta = False
		if (not self.mv): self.mv = -1
		r = self.fstr % (cv, self.mv, cv*100//self.mv, ', '+self.format_speed_eta(cv, self.mv, time.time()-self.started) if (add_speed_eta) else '')
		return r+self.format_bar(cv, self.mv, width-len(r), chars=self.chars, border=self.border)

	@staticmethod
	def format_bar(cv, mv, width, chars=' ▏▎▍▌▋▊▉█', border='│'):
		cv = max(0, min(mv, cv))
		d = 100/(width-2)
		fp, pp = divmod(cv*100/d, mv)
		pb = chars[-1]*int(fp) + chars[int(pp/mv * len(chars))]*(cv != mv)
		return border+(pb+' '*int(width-2-len(pb)))+border

	@staticmethod
	def format_speed_eta(cv, mv, elapsed):
		speed, speed_u = cv/elapsed, 0
		eta = math.ceil(mv/speed)-elapsed if (speed) else 0
		eta = ' '.join(reversed(tuple(str(int(i))+'smhd'[ii] for ii, i in enumerate(S([eta%60, eta//60%60, eta//60//60%24, eta//60//60//24]).strip())))) if (eta > 0) else '?'
		for i in (60, 60, 24):
			if (speed < 1): speed *= i; speed_u += 1
		return '%d/%c, %s ETA' % (speed, 'smhd'[speed_u], eta)

	def print(self, cv, *, out=sys.stderr, width=None, flush=True):
		if (width is None): width = os.get_terminal_size()[0]
		out.write('\033[K'+self.format(cv, width=width)+'\r')
		if (flush): out.flush()
		self.printed = (out == sys.stderr)

class ProgressPool:
	@dispatch
	def __init__(self, *p: Progress):
		self.p = list(p)
		self.ranges = list()

	@dispatch
	def __init__(self, n: int):
		self.__init__(*(Progress() for _ in range(n)))

	def __del__(self):
		for i in self.p: i.printed = False
		sys.stderr.write('\033[J')
		sys.stderr.flush()

	def print(self, *cvs, width=None):
		ii = None
		for ii, (p, cv) in enumerate(zip(self.p, cvs)):
			if (ii): sys.stderr.write('\n')
			p.print(cv, width=width)
		if (ii): sys.stderr.write(f"\033[{ii}A")
		sys.stderr.flush()

	def range(self, start, stop=None, step=1):
		if (stop is None): start, stop = 0, start
		n = len(self.ranges)
		self.ranges.append(int())
		if (n == len(self.p)): self.p.append(Progress())
		self.p[n].mv = stop-start
		for i in range(start, stop, step):
			self.ranges[n] = i-start
			if (n == len(self.p)-1): self.print(*self.ranges)
			yield i
		self.ranges[n] = stop
		self.print(*self.ranges)
		self.ranges.pop()

	@dispatch
	def iter(self, iterator: typing.Iterator, l: int, step: int = 1):
		yield from (next(iterator) for _ in self.range(l, step=step))

	@dispatch
	def iter(self, iterable: typing.Iterable):
		it = tuple(iterable)
		yield from (it[i] for i in self.range(len(it)))

	def done(self, width=None):
		self.print(*(i.mv for i in self.p), width=width)

class ThreadedProgressPool(ProgressPool, threading.Thread):
	def __init__(self, *args, width=None, delay=0.01, **kwargs):
		threading.Thread.__init__(self, daemon=True)
		super().__init__(*args, **kwargs)
		self.width, self.delay = width, delay
		self.stopped = bool()
		self.cvs = [int()]*len(self.p)
		self.ranges = int()

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, type, value, tb):
		self.stop()

	def run(self):
		while (not self.stopped):
			locklog()
			self.print(*self.cvs, width=self.width)
			unlocklog()
			time.sleep(self.delay)

	def range(self, start, stop=None, step=1):
		if (stop is None): start, stop = 0, start
		n = self.ranges
		self.ranges += 1
		if (n == len(self.p)):
			self.p.append(Progress())
			self.cvs.append(int())
		self.p[n].mv = stop-start
		for i in range(start, stop, step):
			self.cvs[n] = i-start
			yield i
		self.ranges -= 1

	def stop(self):
		self.stopped = True
		self.join()
		sys.stderr.write('\r\033[K')
		sys.stderr.flush()

def progrange(start, stop=None, step=1):
	if (stop is None): start, stop = 0, start
	p = Progress(stop-start)
	for i in range(start, stop, step):
		p.print(i-start)
		yield i
	p.print(stop)

@dispatch
def progiter(iterator: typing.Iterator, l: int): # TODO: why yield?
	yield from (next(iterator) for _ in progrange(l))

@dispatch
def progiter(iterable: typing.Iterable):
	l = tuple(iterable)
	yield from (l[i] for i in progrange(len(l)))

def testprogress(n=1000, sleep=0.002):
	p = Progress(n)
	for i in range(n+1):
		p.print(i)
		time.sleep(sleep)

class NodesTree:
	chars = '┌├└─│╾╼'
	colorchars = (*('\033[2m'+i+'\033[22m' for i in chars),)

	def __init__(self, x):
		self.tree = x

	def print(self, out=sys.stdout, **fmtkwargs):
		parseargs(fmtkwargs, color=out.isatty())
		out.write(self.format(**fmtkwargs)+'\n')

	def format(self, **fmtkwargs):
		return '\n'.join(self.format_node(self.tree, root=True, **fmtkwargs))

	@classmethod
	def format_node(cls, node, indent=2, color=False, usenodechars=False, root=False):
		chars = cls.colorchars if (color) else cls.chars
		nodechar = chars[6] if (usenodechars) else chars[3]
		for ii, (k, v) in enumerate(node.items()):
			if (isiterable(v) and not isinstance(v, (str, dict))): k, v = v
			yield ((chars[0] if (len(node) > 1) else chars[5]) if (root and ii == 0) else chars[1] if (ii < len(node)-1) else chars[2])+chars[3]*(indent-1)+nodechar+' '+str(k)
			it = tuple(cls.format_node(v if (isinstance(v, dict)) else {v: {}}, indent=indent, color=color, usenodechars=usenodechars))
			yield from map(lambda x: (chars[4] if (ii < len(node)-1) else ' ')+' '*(indent+1)+x, it)

class TaskTree(NodesTree):
	class Task:
		__slots__ = ('title', 'state', 'subtasks')

		def __init__(self, title, subtasks=None):
			self.title, self.subtasks = str(title), subtasks if (subtasks is not None) else []
			self.state = None

	def __init__(self, x, **kwargs):
		self.tree = x
		self.l = int()
		l = tuple(self.format_node(self.tree, root=True, **kwargs))
		self.l = len(l)

	def __del__(self):
		sys.stderr.write('\n'*self.l)
		sys.stderr.flush()

	def print(self, **fmtkwargs):
		sys.stderr.write('\033[J')
		sys.stderr.write('\n'.join(x+' '+self.format_task(y) for x, y in self.format_node(self.tree, root=True, **fmtkwargs)))
		sys.stderr.write(f"\r\033[{self.l-1}A")
		sys.stderr.flush()

	def format_node(self, node, indent=2, color=False, root=False):
		chars = self.colorchars if (color) else self.chars
		for ii, t in enumerate(node):
			yield (((chars[0] if (len(node) > 1) else chars[5]) if (root and ii == 0) else chars[1] if (ii < len(node)-1) else chars[2])+chars[3]*(indent-1)+(chars[6] if (t.state) else chars[3]), t)
			it = tuple(self.format_node(t.subtasks, indent=indent, color=color))
			yield from map(lambda x: ((chars[4] if (ii < len(node)-1) else ' ')+' '*indent+x[0], x[1]), it)

	@staticmethod
	def format_task(t):
		return f"\033[{93 if (t.state is None) else 91 if (not t.state) else 92}m{t.title}\033[0m"

def validate(l, d, nolog=False):
	for i in d:
		try:
			t, e = d[i] if (type(d[i]) == tuple) else (d[i], 'True')
			assert eval(e.format(t(l[i]))) if (type(e) == str) else e(t(l[i]))
		except Exception as ex:
			if (not nolog): log(2, "\033[91mValidation error:\033[0m %s" % ex)
			return False
	return True

def decline(n, w, prefix='', sep=' ', *, format=False, show_one=True):
	if (isinstance(w, str)): w = (w,)*3
	elif (len(w) == 1): w *= 3
	elif (len(w) == 2): w = (*w, w[-1])

	if (isinstance(prefix, str)): prefix = (prefix,)*3
	elif (len(prefix) == 1): prefix *= 3
	elif (len(prefix) == 2): prefix = (*prefix, prefix[-1])

	if (5 <= abs(n % 100) <= 20): q = 0
	else: q = abs(n) % 10
	if (q == 1): r, p = w[0], prefix[0]
	elif (2 <= q <= 4): r, p = w[1], prefix[1]
	else: r, p = w[2], prefix[2]
	return f"{p}{str(S(n).format(' ' if (format is True) else format) if (format) else n)+sep if (n != 1 or show_one) else ''}{r}"
def testdecline(w): return '\n'.join([decline(i, w) for i in range(10)])

def _timeago(s=-1): # TODO
	if (s == -1): s = time.time()
	s = datetime.timedelta(s)
	s, d = s.seconds, s.days
	s

def frame(x, c=' ', j='.'): # j: {'<', '.', '>'}
	x = x.split('\n')
	w = max(map(len, x))
	return	'╭'+'─'*(w+2)+'╮'+'\n'+\
		'\n'.join('│'+c+S(i).just(w, j)+c+'│' for i in x)+'\n'+\
		'╰'+'─'*(w+2)+'╯'

def iter_queue(q):
	while (q.qsize()): yield q.get()

def first(l): return next(iter(l))

def pm(x): return 1 if (x) else -1
def constrain(x, lb, ub): return min(ub, max(lb, x))
def prod(x, initial=1): return functools.reduce(operator.mul, x, initial)
def average(x, default=None): return sum(x)/len(x) if (x) else default if (default is not None) else raise_(ValueError("average() arg is an empty sequence"))

global_lambdas = list() # dunno why
def global_lambda(l): global_lambdas.append(l); return global_lambdas[-1]

class lc:
	def __init__(self, lc):
		self.lc = lc

	def __enter__(self):
		self.pl = locale.setlocale(locale.LC_ALL)
		locale.setlocale(locale.LC_ALL, self.lc)

	def __exit__(self, type, value, tb):
		locale.setlocale(locale.LC_ALL, self.pl)

class ll:
	def __init__(self, ll):
		self.ll = ll

	def __enter__(self):
		self.pl = loglevel
		setloglevel(self.ll)

	def __exit__(self, type, value, tb):
		setloglevel(self.pl)

class timecounter:
	def __init__(self):
		self.started = None
		self.ended = None

	def __enter__(self):
		self.started = time.time()
		return self

	def __exit__(self, type, value, tb):
		self.ended = time.time()

	def time(self):
		if (self.started is None): return 0
		if (self.ended is None): return time.time()-self.started
		return self.ended-self.started

class classproperty:
	def __init__(self, f):
		self.f = f

	def __get__(self, obj, cls):
		return self.f(cls)

class itemget:
	__slots__ = ('f',)

	class getter:
		__slots__ = ('obj', 'f')

		def __init__(self, obj, f):
			self.obj, self.f = obj, f

		def __getitem__(self, x):
			return self.f(self.obj, *(x if (isinstance(x, tuple)) else (x,)))

	def __init__(self, f):
		self.f = f

	def __get__(self, obj, cls):
		return self.getter(obj, self.f)

class staticitemget:
	__slots__ = ('f', '_fkeys')

	def __init__(self, f):
		self.f = f
		self._fkeys = lambda self: ()

	def __getitem__(self, x):
		return self.f(*x) if (isinstance(x, tuple)) else self.f(x)

	def __contains__(self, x):
		if (x in self.keys()): return True
		try: self[x]
		except KeyError: return False
		else: return True

	def __call__(self, *args, **kwargs):
		return self.f(*args, **kwargs)

	def fkeys(self, f):
		self._fkeys = f
		return f

	def keys(self):
		return self._fkeys(self)

class AttrView(dict):
	def __init__(self, obj):
		self.obj = obj

	def __contains__(self, k):
		if (not isinstance(k, str)): return False
		return hasattr(self.obj, k)

	def __getitem__(self, k):
		if (not isinstance(k, str)): raise KeyError(k)
		return getattr(self.obj, k)

	def __setitem__(self, k, v):
		if (not isinstance(k, str)): raise KeyError(k)
		setattr(self.obj, k, v)

class Builder:
	def __init__(self, cls, *args, **kwargs):
		self.cls = cls
		self.calls = collections.deque(((None, args, kwargs),))

	def __repr__(self):
		return f"<Builder of {repr(self.cls).strip('<>')}>"

	def __getattr__(self, x):
		return lambda *args, **kwargs: self.calls.append((x, args, kwargs)) and None or self

	def build(self):
		instance = None
		for method, args, kwargs in self.calls:
			if (instance is None): instance = self.cls(*args, **kwargs)
			else: getattr(instance, method)(*args, **kwargs)
		return instance

class MetaBuilder(type):
	class Var:
		def __init__(self, name):
			self.name = name

		def __repr__(self):
			return f"<Var '{self.name}'>"

	def __prepare__(name, bases):
		return type('', (dict,), {'__getitem__': lambda self, x: MetaBuilder.Var(x[2:]) if (x.startswith('a_') and x[2:]) else dict.__getitem__(self, x)})()

class SlotsMeta(type):
	def __new__(metacls, name, bases, classdict):
		annotations = dict()
		for i in bases:
			if (hasattr(bases, '__annotations__')): annotations.update(bases.__annotations__)
		if ('__annotations__' in classdict): annotations.update(classdict['__annotations__'])
		classdict['__slots__'] = tuple(annotations.keys())
		if (not annotations): return super().__new__(metacls, name, bases, classdict)
		cls = super().__new__(metacls, name, bases, classdict)
		__init_o__ = cls.__init__
		def __init__(self, *args, **kwargs):
			for k, v in annotations.items():
				if (not hasattr(self, k)): setattr(self, k, v() if (isinstance(v, (type, function))) else v)
			__init_o__(self, *args, **kwargs)
		cls.__init__ = __init__
		#cls.__metaclass__ = metacls  # inheritance?
		return cls

class IndexedMeta(type):
	class Indexer(dict):
		def __init__(self):
			self.l = list()

		def __setitem__(self, k, v):
			if (not k.startswith('__')): self.l.append(v)
			super().__setitem__(k, v)

	@classmethod
	def __prepare__(metacls, name, bases):
		return metacls.Indexer()

	def __new__(metacls, name, bases, classdict):
		cls = super().__new__(metacls, name, bases, dict(classdict))
		cls.__list__ = classdict.l
		#cls.__metaclass__ = metacls  # inheritance?
		return cls

@funcdecorator
def instantiate(f):
	""" Decorator that replaces type returned by decorated function with instance of that type and leaves it as is if it is not a type. """
	def decorated(*args, **kwargs):
		r = f(*args, **kwargs)
		return r() if (isinstance(r, type)) else r
	return decorated

@dispatch
def singleton(C: type): return C()
@dispatch
def singleton(*args, **kwargs): return lambda C: C(*args, **kwargs)

@cachedfunction
def getsubparser(ap, *args, **kwargs): return ap.add_subparsers(*args, **kwargs)

#@dispatch
@funcdecorator
def apmain(f, nolog=False):
	def decorated():
		if (hasattr(f, '_argdefs')):
			while (f._argdefs):
				args, kwargs = f._argdefs.popleft()
				argparser.add_argument(*args, **kwargs)
		argparser.set_defaults(func=lambda *_: sys.exit(argparser.print_help()))
		cargs = argparser.parse_args()
		if (not nolog): logstarted()
		return f(cargs)
	return decorated

def apcmd(*args, **kwargs):
	@funcdecorator
	def decorator(f):
		nonlocal args, kwargs
		if (not hasattr(f, '_argdefs')): raise ValueError(f"Function decorated with @apcmd should also be decorated with @aparg")
		subparser = getsubparser(argparser, *args, **kwargs).add_parser(f.__name__.rstrip('_'), help=f.__doc__)
		for args, kwargs in f._argdefs:
			subparser.add_argument(*args, **kwargs)
		subparser.set_defaults(func=f._f)
		return f._f
	return decorator

def aparg(*args, **kwargs):
	@funcdecorator
	def decorator(f):
		if (not hasattr(f, '_argdefs')):
			of = f
			def f(cargs):
				for args, kwargs in f._argdefs:
					argparser.add_argument(*args, **kwargs)
				cargs = argparser.parse_args()
				return of(cargs)
			f._argdefs = collections.deque()
			f._f = of
		f._argdefs.appendleft((args, kwargs))
		return f
	return decorator

class hashabledict(dict):
	__setitem__ = \
	__delitem__ = \
	clear = \
	pop = \
	popitem = \
	setdefault = \
	update = classmethod(lambda cls, *args, **kwargs: raise_(TypeError(f"'{cls.__name__}' object is not modifiable")))

	def __hash__(self):
		return hash(tuple(self.items()))

@dispatch
def tohashable(d: typing.Dict): return tuple(sorted((k, tohashable(v)) for k, v in d.items()))
@dispatch
def tohashable(s: str): return s
@dispatch
def tohashable(l: typing.Iterable): return tuple(map(tohashable, l))
@dispatch
def tohashable(x: typing.Hashable): return x

class paramset(set):
	def __init__(self, x=()):
		super().__init__(map(str, x))

	def __hash__(self):
		return hash(tuple(sorted(self)))

	def __getattr__(self, x):
		return str(x) in self
	__getitem__ = __getattr__

	def __setattr__(self, x, y):
		if (y): self.add(str(x))
	__setitem__ = __setattr__

	def __delattr__(self, x):
		self.discard(str(x))
	__delitem__ = __delattr__

class listmap:
	def __init__(self):
		self._keys = collections.deque()
		self._values = collections.deque()

	def __getitem__(self, k):
		return self._values[self._keys.index(k)]

	def __setitem__(self, k, v):
		if (k in self._keys): self._values[self._keys.index(k)]
		else:
			self._keys.append(k)
			self._values.append(v)

	def __contains__(self, k):
		return k in self._keys

	def __iter__(self):
		return iter(self._keys)

	def index(self, x):
		return self._keys.index(x)

	def keys(self):
		return self._keys

	def values(self):
		return self._values

	def items(self):
		for k in self.keys():
			yield (k, self[k])

class indexset:
	def __init__(self):
		self._list = list()

	def __getitem__(self, x):
		try: return self._list.index(x)
		except ValueError:
			self._list.append(x)
			return len(self._list)-1

	def __delitem__(self, x):
		self._list.remove(x)

	@itemget
	def values(self, x):
		return self._list[x]

#class fields(abc.ABC):
#	""" A collections.namedtuple-like structure with attribute initialization and type casting.
#	Based on init_defaults() and autocast().
#	You should declare subclasses as:
#	class C(fields):
#		__fields__ = {'a': None, 'b': int}
#	"""
#
#	__fields__ = type('__fields__', (), {'__isabstractmethod__': True})()
#
#	def __init__(self, **kwargs):
#		for k, v in self.__fields__.items():
#			setattr(self, k, kwargs[k])

def getsrc(x, clear_term=True):
	if (clear_term): clear()
	print(inspect.getsource(x))

def preeval(f):
        r = f()
        return lambda: r

def printf(format, *args, file=sys.stdout, flush=False): print(format % args, end='', file=file, flush=flush)
class _CStream: # because I can.
	@dispatch
	def __init__(self, fd):
		self.ifd = self.ofd = fd

	@dispatch
	def __init__(self, ifd, ofd):
		self.ifd = ifd
		self.ofd = ofd

	def __repr__(self):
		return '\033[F' if (sys.flags.interactive) else super().__repr__()
class IStream(_CStream):
	def __rshift__(self, x):
		globals = inspect.stack()[1][0].f_globals
		globals[x] = type(globals.get(x, ''))(self.ifd.readline().rstrip('\n')) # obviousity? naaooo
		return self
class OStream(_CStream):
	def __lshift__(self, x):
		self.ofd.write(str(x))
		if (x is endl): self.ofd.flush()
		return self
class IOStream(IStream, OStream): pass
cin = IStream(sys.stdin)
cout = OStream(sys.stdout)
cerr = OStream(sys.stderr)
cio = IOStream(sys.stdin, sys.stdout)

class UserError(Exception): pass
class WTFException(AssertionError): pass
class TODO(NotImplementedError): pass
class TEST(BaseException): pass
class NonLoggingException(Exception): pass

def exit(c=None, code=None, raw=False, nolog=False):
	sys.stderr.write('\r\033[K')
	unlocklog()
	db.save(nolog=True)
	if (not nolog): log(f'{c}' if (raw) else f'Exit: {c}' if (c and type(c) == str or hasattr(c, 'args') and c.args) else 'Exit.')
	try: sys.exit(int(bool(c)) if (code is not None) else code)
	finally:
		if (not nolog): log(raw=True)

def setonsignals(f=exit): signal.signal(signal.SIGINT, f); signal.signal(signal.SIGTERM, f)

logstart('Utils')
if (__name__ == '__main__'):
	testprogress()
	log('\033[0mWhy\033[0;2m are u trying to run me?! It \033[0;93mtickles\033[0;2m!..\033[0m', raw=True)
else: logimported()

# by Sdore, 2019
