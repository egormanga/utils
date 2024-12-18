#!/usr/bin/env python3
# Utils lib
#type: ignore

""" Sdore's Utils

Code example:

from utils import *; logstart('<NAME>')

def main(): pass

if (__name__ == '__main__'): logstarted(); main()
else: logimported()

"""

import sys, queue, regex, types, inspect, argparse, pygments, readline, warnings, typing_inspect

py_version = f"Python {sys.version.split(maxsplit=1)[0]}"
__repl__ = bool(getattr(sys, 'ps1', sys.flags.interactive))

class ModuleProxy(types.ModuleType):
	def __init__(self, name):
		self.__name__ = name

	def __getattribute__(self, x):
		module = __import__(name := object.__getattribute__(self, '__name__'))
		try: inspect.stack(0)[1].frame.f_globals[name] = module
		except (TypeError, IndexError): pass
		if (module.__name__ not in {*getattr(sys, 'stdlib_module_names', ()), *sys.builtin_module_names} or
		    any('site-packages' in (getattr(module, i, None) or '') for i in ('__file__', '__path__'))):
			warnings.warn(f"Using non-standard modules ({module}) through utils is deprecated.", stacklevel=2)
		return getattr(module, x)

_globals = globals()
_globals.update(sys.modules)  # no import cost at all, as those are already loaded
_autoimports = (
	're',
	'ast',
	'cmd',
	'dis',
	'pdb',
	'pty',
	'tty',
	'code',
	'copy',
	'glob',
	'gzip',
	'hmac',
	'html',
	'http',
	'json',
	'math',
	'uuid',
	'zlib',
	'errno',
	'shlex',
	'atexit',
	'base64',
	'bisect',
	'codeop',
	'ctypes',
	'locale',
	'pickle',
	'pprint',
	'random',
	'select',
	'shutil',
	'signal',
	'socket',
	'string',
	'struct',
	'typing',
	'urllib',
	'asyncio',
	'getpass',
	'hashlib',
	'numbers',
	'secrets',
	'termios',
	'zipfile',
	'calendar',
	'datetime',
	'platform',
	'tempfile',
	'fractions',
	'ipaddress',
	'linecache',
	'mimetypes',
	'threading',
	'traceback',
	'subprocess',
	'rlcompleter',
	'unicodedata',
	'configparser',
	'socketserver',
	'multiprocessing',
	#'nonexistenttest'
)

_globals_ = frozenset(sum((tuple(i.frame.f_globals) for i in inspect.stack(0)[1:]), start=()))
#print(_globals_)
for i in _autoimports:
	if (i in _globals_): continue
	if (i in _globals):
		#warnings.warn(f"redundant autoimport: {i}")
		continue
	_globals[i] = ModuleProxy(i)
del i, _globals

def install_all_imports():
	r = list()
	for i in _autoimports:
		i = i.partition(' ')[0]
		if (i in sys.modules): continue
		try: __import__(i)
		except ModuleNotFoundError: r.append(i)
	old_sysargv, sys.argv = sys.argv, ['pip3', 'install', *r]
	try: __import__('pkg_resources').load_entry_point('pip', 'console_scripts', 'pip3')()
	finally: sys.argv = old_sysargv

argparser = argparse.ArgumentParser(conflict_handler='resolve', add_help=False)
argparser.add_argument('-v', action='count', help=argparse.SUPPRESS)
argparser.add_argument('-q', action='count', help=argparse.SUPPRESS)
cargs = argparser.parse_known_args()[0]
argparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
loglevel = ((cargs.v or 0) - (cargs.q or 0))

# TODO: types.*
dict_keys, dict_values, dict_items = type({}.keys()), type({}.values()), type({}.items())
module = types.ModuleType
function = types.FunctionType
builtin_function_or_method = types.BuiltinFunctionType
method = types.MethodType
method_descriptor = types.MethodDescriptorType
generator = types.GeneratorType
coroutine = types.CoroutineType
async_generator = types.AsyncGeneratorType
CodeType = types.CodeType
TracebackType = types.TracebackType
NoneType = type(None)
inf = float('inf')
nan = float('nan')
nl = endl = NL = ENDL = '\n'
tab = TAB = '\t'

def isiterable(x): return isinstance(x, typing.Iterable)
def isiterablenostr(x): return isiterable(x) and not isinstance(x, str)
def isnumber(x): return isinstance(x, numbers.Number)
def parseargs(kwargs, **args): args |= kwargs; kwargs |= args; return kwargs
def setdefault(d: dict, **kwargs) -> dict: r = d | kwargs; r |= d; return r
def hexf(x, l=2): return (f"0x%0{l}X" % x)
def md5(x, n=1): x = x if (isinstance(x, bytes)) else str(x).encode(); return hashlib.md5(md5(x, n-1).encode() if (n > 1) else x).hexdigest()
def b64(x): return base64.b64encode(x if (isinstance(x, bytes)) else str(x).encode()).decode()
def ub64(x): return base64.b64decode(x).decode()
def randstr(n=16, *, caseless=False, seed=None): return str().join((random.Random(seed) if (seed is not None) else random).choices(string.ascii_lowercase if (caseless) else string.ascii_letters, k=n))
def try_repr(x) -> str:
	try: return repr(x)
	except Exception: return object.__repr__(x)
def try_eval(*args, **kwargs):
	try: return eval(*args, **kwargs)
	except Exception: return None
def safeexec():
	try:
		print(end='\033[2m>>> ', file=sys.stderr, flush=True)
		return exec(sys.stdin.readline())
	finally: print(end='\033[m', file=sys.stderr, flush=True)
def export(x):
	globals = inspect.stack(0)[1].frame.f_globals
	all = globals.setdefault('__all__', [])
	if (not isinstance(all, list)): all = globals['__all__'] = list(all)
	try: name = x.__qualname__
	except AttributeError:
		try: name = x.__name__
		except AttributeError: name = x.__class__.__name__
	name = name.rpartition('.')[-1]
	if (name not in all): all.append(name)
	return x
def suppress_tb(f):
	f.__code__ = code_with(f.__code__, name=f.__qualname__, firstlineno=0, **{'linetable' if (sys.version_info >= (3, 10)) else 'lnotab': (b'\xff'*(len(f.__code__.co_code)//2+1) if (sys.version_info >= (3, 11)) else b'')})
	return f

def code_with(code, **kwargs):
	if (sys.version_info >= (3, 8)): return code.replace(**{'co_'*(not k.startswith('co_'))+k: v for k, v in kwargs.items()})
	return CodeType(*(kwargs.get(i, kwargs.get('co_'+i, getattr(code, 'co_'+i))) for i in ('argcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', 'lnotab', 'freevars', 'cellvars')))
# TODO: func_with()

def terminal_link(link, text=None): return f"\033]8;;{Sstr(link).noesc().fit(1634, bytes=True)}\033\\{text if (text is not None) else link}\033]8;;\033\\"

import pygments.lexers, pygments.formatters
def highlight(s: str, *, lexer=pygments.lexers.PythonLexer(), formatter=pygments.formatters.TerminalFormatter(bg='dark')): return pygments.highlight(s, lexer, formatter)

def S(*x, ni_ok=False):
	""" Convert `x' to an instance of corresponding S-type. """

	r = tuple(i if (isinstance(i, S_type)) else Stuple(i) if (isinstance(i, generator)) else globals().get('S'+i.__class__.__name__, lambda i: raise_(NotImplementedError('S'+i.__class__.__name__)) if (not ni_ok) else i)(i) for i in x)
	return (first(r) if (len(r) == 1) else r)

class S_type: pass

class Sdict(S_type, collections.defaultdict):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if (not args or isiterable(args[0])): args.insert(0, None)
		super().__init__(*args, **kwargs)

	__repr__ = dict.__repr__

	def __getattr__(self, x):
		try: r = self[x]
		except KeyError: pass
		else: return Sdict(r) if (isinstance(r, dict)) else r
		raise AttributeError(x)

	def __setattr__(self, x, v):
		try: self[x] = v
		except KeyError: pass
		else: return
		raise AttributeError(x)

	@suppress_tb
	def __missing__(self, k):
		if (self.default_factory is None): raise KeyError(k)
		try: r = self.default_factory()
		except TypeError:
			try: r = self.default_factory(k)
			except TypeError: ok = False
			else: ok = True
			if (not ok): raise
		self[k] = r
		return r

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

	def translate(self, table, *, copy=False, strict=True, keep=True):
		r = Sdict(self)
		for i, v in table.items():
			k, t = v if (isinstance(v, (tuple, list))) else (v, lambda x: x)
			if (not strict and k not in r): continue
			if (keep and i not in r): r[i] = t((r.get if (copy) else r.pop)(k))
		return r

	def with_(self, key, value=None):
		""" Return self with item `key' set to `value'. """
		r = self.copy()
		r[key] = value
		return r

	def filter(self, value):
		""" Return self without items with value equal to `value'. """
		return Sdict({k: v for k, v in self.items() if v != value})

	def filter_exact(self, value):
		""" Return self without items with value `value' (by `is not' operator). """
		return Sdict({k: v for k, v in self.items() if v is not value})

	def filter_bool(self):
		""" Return self without items for which `bool(value)' evaluates to `False'. """
		return Sdict({k: v for k, v in self.items() if v})

	_to_discard = set()

	def to_discard(self, x):
		self._to_discard.add(x)

	def discard(self):
		for i in self._to_discard:
			#try:
			self.pop(i)
			#except IndexError: pass
		self._to_discard.clear()
		return self
Sdefaultdict = Sdict

class Slist(S_type, list): # TODO: fix everywhere: type(x) == y --> isinstance(x, y)
	def __matmul__(self, item):
		if (type(item) == dict): return Slist(i for i in self if all(v(i.get(k)) if (callable(v)) else i.get(k) in v for k, v in item.items()))
		r = Slist(Slist((i.get(j) if (hasattr(i, 'get')) else i[j]) for j in item) for i in self)
		return r.flatten() if (len(item) == 1 and not isiterable(item[0]) or type(item[0]) == str) else r # TODO FIXME isiterablenostr?

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

	def rindex(self, x, start=0): # TODO end
		return len(self)-self[:(start or None) and start-1:-1].index(x)-1

	def group(self, n):
		return Slist(tuple(j for j in i if j is not None) for i in itertools.zip_longest(*(iter(self),)*n))

	def flatten(self):
		return Slist(j for i in self if i for j in (i if (isiterablenostr(i)) else (i,)))

	def strip(self, s=None):
		l = self.copy()
		if (not isiterable(s) or isinstance(s, str)): s = (s,)
		return Slist(i for i in l if (i if (s is None) else i not in s))

	def filter(self, value):
		""" Return self without items equal to `value'. """
		return Slist(i for i in self if i != value)

	def filter_bool(self):
		""" Return self without items for which `bool(value)' evaluates to `False'. """
		return Sdict({k: v for k, v in self.items() if v})

	def filter_isinstance(self, type):
		""" Return self with only items of type `type'. """
		return Slist(i for i in self if isinstance(i, type))

	def uniquize(self, key=None):
		was = list()
		return Slist(was.append(key(i) if (key is not None) else i) or i for i in self if (key(i) if (key is not None) else i) not in was)  # such a dirty hack.. upd. even more dirty for unhashable

	#def wrap(self,

	_to_discard = set()
	def to_discard(self, ii):
		self._to_discard.add(ii)

	def discard(self):
		to_del = [self[ii] for ii in self._to_discard]
		for i in to_del:
			#try:
			self.remove(i)
			#except IndexError: pass
		self._to_discard.clear()
		return self

class Stuple(Slist): pass # TODO

class Sint(S_type, int):
	def __len__(self):
		return Sint(math.log10(abs(self) or 1)+1)

	def constrain(self, lb, ub):
		return Sint(constrain(self, lb, ub))

	def format(self, char=' '):
		return char.join(map(str().join, Slist(str(self)[::-1]).group(3)))[::-1]

	def pm(self):
		return f"+{self}" if (self > 0) else str(self)

class Sstr(S_type, str):
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

	def __or__(self, x):
		return self if (self.strip()) else x

	def group(self, n, sep=' '):
		return S(sep).join(str().join(j for j in i if j is not None) for i in itertools.zip_longest(*(iter(self),)*n))

	def fit(self, l, *, end='…', noesc=True, bytes=False):
		if (noesc): self = self.noesc()
		if (bytes):
			enc = self.encode()
			end_enc = end.encode()
			return Sstr(self if (len(enc) <= l) else enc[:l-len(end_enc)].decode(errors='ignore')+end if (l >= len(end_enc)) else '')
		else: return Sstr(self if (len(self) <= l) else self[:l-len(end)]+end if (l >= len(end)) else '')

	def cyclefit(self, l, n, *, sep=' '*8, start_delay=0, **kwargs):
		if (len(self) <= l): return self
		n = max(0, (n % (len(self) + len(sep) + start_delay)) - start_delay)
		return Sstr((self + sep)[n:] + self[:n]).fit(l)

	def join(self, l, *, first='', last=None):
		l = tuple(map(str, l))
		r = (str.join(self, l[:-1]) + ((last if (isinstance(last, str)) else last[len(l) > 2]) if (last is not None) else self) + l[-1]) if (len(l) > 1) else l[0] if (l) else ''
		if (r): r = first+r
		return Sstr(r)

	def bool(self, minus_one=True):
		return bool(self) and self.casefold() not in ('0', 'false', 'no', 'нет', '-1'*(not minus_one))

	def indent(self, n=None, char=None, *, tab_width=8, foldempty=True, first=True):
		if (not self): return self
		if (n is None): n = tab_width
		r, n = n < 0, abs(n)
		if (char is None): char = ('\t'*(n//tab_width)+' '*(n % tab_width)) if (not r) else ' '*n
		else: char *= n
		res = char if (first) else ''
		if (not r): res = res+('\n'+char).join(self.split('\n'))
		else:       res = (char+'\n').join(self.split('\n'))+res
		if (foldempty): res = re.sub(r'^\s+?$', '', res, flags=re.M)
		return Sstr(res)

	def unindent(self, n=None, char='\t', *, skipempty=True):
		if (n is None): n = min(Sstr(i).lstripcount(char)[0] for i in self.split('\n') if not skipempty or i.strip())
		return Sstr(re.sub(r'^'+char*n, '', self, flags=re.M)) if (n > 0) else self

	def lstripcount(self, chars=None):
		return lstripcount(self, chars=chars)

	def just(self, n, char=' ', j=None):
		if (j is None): j, n = '<>'[n>0], abs(n)
		if (j == '.'): r = self.center(n, char)
		elif (j == '<'): r = self.ljust(n, char)
		elif (j == '>'): r = self.rjust(n, char)
		else: raise ValueError(j)
		return Sstr(r)

	def sjust(self, n, *args, **kwargs):
		return self.indent(n-len(self), *args, **kwargs)

	def wrap(self, w, char=' ', loff=0, sep=' '):
		w -= loff
		if (len(self) <= w): return self
		r, *s = self.split(sep)
		cw = w-len(r)
		for i in s:
			if (cw >= len(sep)+len(i)): r += sep; cw -= len(sep)
			else: r += '\n'; cw = w
			r += i
			cw -= len(i)
			#i = i.replace('\r', '\n') # TODO?
			if ('\n' in i): cw = w-len(i[i.rindex('\n'):])
		return Sstr('\n'.join(char*(loff*bool(ii))+i for ii, i in enumerate(r.rstrip(' ').split('\n'))))

	def split(self, *args, **kwargs):
		return Slist(map(Sstr, str.split(self, *args, **kwargs)))

	def filter(self, chars):
		return Sstr().join(i for i in self if i in chars)

	def noesc(self):
		return Sstr(noesc.sub('', self))

	def capwords(self, sep=' '):
		s = list(self)
		c = True
		for ii, i in enumerate(s):
			if (c): s[ii] = i.upper(); c = False
			if (i in sep): c = True
		return Sstr().join(s)

	def fullwidth(self):
		return sum(c.isprintable() and not unicodedata.combining(c) for c in self)

	def sub(self):
		return self.translate(self._subtrans)

	def super(self): logexception(DeprecationWarning(" *** super() → sup() *** ")); return self.sup()

	def sup(self):
		return self.translate(self._suptrans)

def Sbool(x=bool(), *args, **kwargs): # No way to derive a class from bool
	x = S(x)
	return x.bool(*args, **kwargs) if (hasattr(x, 'bool')) else bool(x)

def funcdecorator(df=None, /, *, signature=None, suppresstb=True): # TODO: __dict__?
	if (df is None): return lambda df: funcdecorator(df, signature=signature, suppresstb=suppresstb)

	@suppress_tb
	def ndf(f, *args, **kwargs):
		if (not isinstance(f, function)): return lambda nf: ndf(nf, f, *args, **kwargs)

		nf = df(f)
		if (nf is f): return f

		if (suppresstb): nf = suppress_tb(nf) # TODO: option to disable

		nf.__name__, nf.__qualname__, nf.__module__, nf.__doc__, nf.__annotations__ = \
		 f.__name__,  f.__qualname__,  f.__module__,  f.__doc__,  f.__annotations__ # TODO: .copy()?

		if (signature is not None):
			if (isinstance(signature, str)): nf.__text_signature__ = signature
			else: nf.__signature__ = signature
			inspect.signature(nf)  # validate
		else: nf.__signature__ = inspect.signature(f)

		nf.__code__ = code_with(nf.__code__, name=f"<decorated {f.__code__.co_name}>" if (not f.__code__.co_name.startswith('<decorated ')) else f.__code__.co_name)

		for i in filter(lambda x: not x.startswith('__'), dir(f)):
			setattr(nf, i, getattr(f, i))

		nf.__wrapped__ = f

		return nf

	#dfsig, ndfsig = inspect.signature(df), inspect.signature(ndf)
	#if (dfsig != ndfsig): raise ValueError(f"Function decorated with @funcdecorator should have signature '{format_inspect_signature(ndfsig)}' (got: '{format_inspect_signature(dfsig)}')") # TODO kwargs

	ndf.__name__, ndf.__qualname__, ndf.__module__, ndf.__doc__ = \
	 df.__name__,  df.__qualname__,  df.__module__,  df.__doc__

	ndf.__code__ = code_with(ndf.__code__, name=df.__code__.co_name)

	for i in filter(lambda x: not x.startswith('__'), dir(df)):
		setattr(nff, i, getattr(df, i))

	return ndf

class DispatchError(TypeError): pass

class dispatch:
	""" Decorator which implements function overloading (call dispatching) by argument and return type annotations. """

	class __FillValue: __slots__ = ()
	__FillValue = __FillValue()

	class __Param(inspect.Parameter):
		def __repr__(self):
			return f"<Parameter '{self.name}: {format_inspect_annotation(self.type)}' ({self.kind}, {'default' if (not self.required) else 'required'})>"

		@property
		def type(self):
			return self.annotation

		@property
		def required(self):
			return (self.default is inspect._empty)

		@classmethod
		def from_inspect_parameter(cls, param):
			return cls(name=param.name, kind=param.kind, default=param.default, annotation=param.annotation)

	__slots__ = ('__call__', '__qualname__', '__origname__', '__origmodule__')
	__overloaded_functions = Sdict(dict)
	__overloaded_functions_docstrings = Sdict(dict)

	def __init__(self, f):
		self.__origname__, self.__origmodule__ = f.__name__, f.__module__
		self.__qualname__ = f.__qualname__
		self.__call__ = method(function(code_with(self.___call__.__code__, name=f"<overload handler of {self.__qualname__}>"), self.___call__.__globals__), self)

		fsig = _inspect_signature(f, eval_str=True)
		params_annotation = tuple(map(self.__Param.from_inspect_parameter, fsig.parameters.values()))

		self.__overloaded_functions[self.__origmodule__, self.__qualname__][params_annotation, fsig.return_annotation] = f
		self.__overloaded_functions_docstrings[self.__origmodule__, self.__qualname__][fsig] = f.__doc__

		wf = inspect.unwrap(f)

		f.__origname__, f.__name__ = f.__name__, f"Overloaded {f.__name__}"
		wf.__code__ = code_with(wf.__code__, name=f"<overloaded {f.__qualname__} for {f.__origname__}{format_inspect_signature(fsig)}>")

		#print(f"\n\033[1;92m{self.__qualname__}()\033[0;1m:\033[m")
		#for params, retval in self.__overloaded_functions[self.__origmodule__, self.__qualname__]:
		#	print("     \N{bullet}", end='')
		#	for param in params:
		#		print(f"\t\033[1;93m{param.name}\033[m: \033[1;94m{format_inspect_annotation(param.type)}\033[m\n\t  \033[2m({param.kind.name}, {'default' if (not param.required) else 'required'})\033[m\n")
		#	print()

	def __repr__(self):
		return f"<overloaded function {self.__qualname__}>"

	@suppress_tb
	def ___call__(self, *args, **kwargs):
		args, kwargs, f, params, retval = self.__dispatch(*args, **kwargs)

		r = f(*args, **kwargs)

		if (retval is not inspect._empty):
			if (not self.__typecheck(r, retval)): raise DispatchError(f"Return value of type {type(r)} doesn't match the return annotation of the corresponding '{self.__qualname__}' signature:\n  {self.__origname__}{_format_inspect_signature({i.name: i for i in params}, retval, _call_lambdas=True)}\n[called as: {self.__origname__}({', '.join((*(S(try_repr(i)).fit(32) for i in args), *(S(f'{k}={try_repr(v)}').fit(32) for k, v in kwargs.items())))})]")

		return r

	def __get__(self, obj, objcls):
		return (method(self, obj) if (obj is not None) else self)

	def __getitem__(self, *t):
		return self.__overloaded_functions[self.__origmodule__, self.__qualname__][t]

	@suppress_tb
	def __dispatch(self, *args, **kwargs):
		for (params, retval), func in self.__overloaded_functions[self.__origmodule__, self.__qualname__].items():
			fsig = _inspect_signature(func, eval_str=True)

			for args_ in (args, args[1:]) if (isinstance(func, staticmethod)) else (args,):
				try: bound = fsig.bind(*args_, **kwargs)
				except TypeError: continue

				pars = {i.name: i for i in params}

				ok = bool()
				for k, v in bound.arguments.items():
					try: p = pars[k]
					except KeyError: break

					if (p.kind is inspect.Parameter.VAR_KEYWORD):
						if (not all(self.__typecheck(i, p.type) for i in v.values())): break
					elif (p.kind is inspect.Parameter.VAR_POSITIONAL):
						if (not all(self.__typecheck(i, p.type) for i in v)): break
					else:
						if (not self.__typecheck(v, p.type)): break
				else: ok = True
				if (not ok): continue

				bound.apply_defaults()

				return (args_, kwargs, func, params, retval)

		try: return (self.__overloaded_functions[self.__origmodule__, self.__qualname__][(), inspect._empty], (), inspect._empty)
		except KeyError: pass

		sigs = self.format_signatures(sep='\n').split('\n')
		raise DispatchError(f"Parameters {S(', ').join((*(format_inspect_annotation(type(i)) for i in args), *(f'{k}: {format_inspect_annotation(type(v))}' for k, v in kwargs.items()))).join('()')} don't match {'any of' if (len(sigs) > 1) else 'the'} '{self.__qualname__}' signature{'s'*(len(sigs) > 1)}:\n{'  • '+f'{NL}  • '.join(sigs) if (len(sigs) > 1) else f'    {only(sigs)}'}\n[called as: {self.__origname__}({', '.join((*(S(try_repr(i)).fit(32) for i in args), *(S(f'{k}={try_repr(v)}').fit(32) for k, v in kwargs.items())))})]")

	@classmethod
	def __typecheck(cls, o, t) -> bool:
		if (getattr(cls, '_dispatch__typecheck_frame', None) is not None): cls.__typecheck_frame = inspect.stack()[0].frame

		if (t is None or t is inspect._empty): return True
		if (o is cls.__FillValue or t is cls.__FillValue): return False

		if (isinstance(t, function) and t.__code__.co_argcount == 0): return cls.__typecheck(o, t())
		if (isinstance(t, (function, builtin_function_or_method, method_descriptor))): return bool(t(o))

		origin, args = typing.get_origin(t), typing.get_args(t)

		if (isinstance(t, typing.TypeVar)):
			if (not isinstance(o, typing_inspect.get_constraints(t))): return False
		elif (typing_inspect.is_literal_type(t)):
			return (o in args)
		elif (typing_inspect.is_union_type(t)):
			if (not any(cls.__typecheck(o, i) for i in args)): return False
			else: return True
		elif (not args):
			if (not isinstance(o, (origin or t))): return False

		if (args):
			if (issubclass(origin, type)):
				tt = only(args)
				tto = (typing.get_origin(tt) or tt)
				oo = (typing.get_origin(o) or o)
				if (not (oo is tto or isinstance(oo, type) and issubclass(oo, (typing.get_args(tt) if (typing_inspect.is_union_type(tt)) else tto)))): return False
				if (typing_inspect.is_generic_type(tt)):
					if (not all(itertools.starmap(lambda a, b: a and b and any(issubclass(i, b) for i in (typing.get_args(a) if (typing_inspect.is_union_type(a)) else a)), itertools.zip_longest(*map(typing.get_args, (o, tt)))))): return False
			elif (issubclass(origin, typing.Tuple) and isinstance(o, typing.Tuple)):
				args = list(args)
				try: ei = args.index(...)
				except ValueError: pass
				else: args[ei:ei+1] = (((args[ei-1] if (ei > 0) else None),)*(len(o) - len(args) + 1) if (len(args) >= 2) else (cls.__FillValue,))
				if (not all(itertools.starmap(cls.__typecheck, itertools.zip_longest(o, args, fillvalue=cls.__FillValue)))): return False
			elif (isinstance(o, typing.Iterable) and not isinstance(o, typing.Iterator)):
				if (not all(cls.__typecheck(i, args[0]) for i in o)): return False
			else: return False # TODO FIXME?

		return True

	def format_signatures(self, sep='\n\n'):
		return sep.join(Sstr().join((self.__qualname__, format_inspect_signature(fsig, _call_lambdas=True), ':\n\b    '+doc if (doc) else '')) for fsig, doc in self.__overloaded_functions_docstrings[self.__origmodule__, self.__qualname__].items() if fsig is not None)

	#@funcdecorator
	@classmethod
	def meta(cls, f):
		if (f.__doc__): cls.__overloaded_functions_docstrings[f.__module__, f.__qualname__][None] = f.__doc__
		return f

	@property
	def __name__(self):
		return f"Overloaded {self.__origname__}"

	@property
	def __doc__(self):
		signatures = self.format_signatures()
		try: doc = self.__overloaded_functions_docstrings[self.__origmodule__, self.__qualname__][None]
		except KeyError: return signatures
		else: return (doc + '\n\n' + signatures)

	@property
	def __signature__(self):
		return ...

	@property
	def signatures(self):
		return tuple(self.__overloaded_functions_docstrings[self.__origmodule__, self.__qualname__].keys())

def dispatch_meta(f): logexception(DeprecationWarning("*** @dispatch_meta → @dispatch.meta ***")); return dispatch.meta(f)

def format_inspect_signature(fsig, *, _call_lambdas=False): return _format_inspect_signature(fsig.parameters, fsig.return_annotation, _call_lambdas=_call_lambdas)
def _format_inspect_signature(parameters, return_annotation=inspect._empty, *, _call_lambdas=False):
	result = list()
	posonlysep = False
	kwonlysep = True

	for p in parameters.values():
		if (p.kind == inspect.Parameter.POSITIONAL_ONLY): posonlysep = True
		elif (posonlysep): result.append('/'); posonlysep = False
		if (p.kind == inspect.Parameter.VAR_POSITIONAL): kwonlysep = False
		elif (p.kind == inspect.Parameter.KEYWORD_ONLY and kwonlysep): result.append('*'); kwonlysep = False
		result.append(f"{'*' if (p.kind == inspect.Parameter.VAR_POSITIONAL) else '**' if (p.kind == inspect.Parameter.VAR_KEYWORD) else ''}{p.name}{f': {format_inspect_annotation(p.annotation, _call_lambdas=_call_lambdas)}' if (p.annotation is not inspect._empty) else ''}{f' = {repr(p.default)}' if (p.annotation is not inspect._empty and p.default is not inspect._empty) else f'={repr(p.default)}' if (p.default is not inspect._empty) else ''}")

	if (posonlysep): result.append('/')
	rendered = ', '.join(result).join('()')
	if (return_annotation is not inspect._empty): rendered += f" -> {format_inspect_annotation(return_annotation)}"
	return rendered

def format_inspect_annotation(annotation, *, _call_lambdas=False):
	if (isinstance(annotation, function)):
		if (_call_lambdas and annotation.__code__.co_argcount == 0):
			try: return format_inspect_annotation(annotation())
			except NameError: pass
		return annotation.__name__
	if (isinstance(annotation, tuple)): return ', '.join(map(format_inspect_annotation, annotation)).join('()')
	return inspect.formatannotation(annotation)

def _inspect_signature(*args, **kwargs) -> inspect.Signature:
	if (sys.version_info < (3, 10)): kwargs.pop('eval_str', None)
	return inspect.signature(*args, **kwargs)

def _inspect_get_annotations(x, *, globals=None, locals=None, eval_str=False):
	try: _get_annotations = inspect.get_annotations
	except AttributeError: pass
	else: return _get_annotations(x, globals=globals, locals=locals, eval_str=eval_str)

	try: res = x.__annotations__.copy()
	except AttributeError: return {}
	if (eval_str):
		for k, v in res.items():
			try: res[k] = eval(v, globals, locals)
			except Exception: pass
	return res

@dispatch
def get_annotations(classdict: lambda x: isinstance(x, dict) and x.get('__module__') in sys.modules, *, eval_str=True, globals=None, locals=None): return get_annotations(classdict.get('__annotations__', {}), eval_str=eval_str, globals=globals if (globals is not None or not eval_str) else getattr(sys.modules.get(classdict['__module__']), '__dict__', None), locals=locals if (locals is not None or not eval_str) else classdict)
@dispatch
def get_annotations(annotations: dict, *, eval_str=False, globals=None, locals=None): return {k: e if (eval_str and isinstance(v, str) and not v.strip().startswith('#') and (e := try_eval(v.split('--')[0], globals, locals))) else v for k, v in annotations.items()}
@dispatch
def get_annotations(x: lambda x: hasattr(x, '__wrapped__'), **kwargs): return get_annotations(x.__wrapped__, **kwargs)
@dispatch
def get_annotations(cls: type, *, eval_str=True, globals=None, locals=None): return get_annotations(_inspect_get_annotations(cls), eval_str=eval_str, globals=globals if (globals is not None or not eval_str) else getattr(sys.modules.get(cls.__module__, {}), '__dict__', None), locals=locals if (locals is not None or not eval_str) else dict(vars(cls)))
@dispatch
def get_annotations(m: module, *, eval_str=True, globals=None, locals=None): return get_annotations(_inspect_get_annotations(m), eval_str=eval_str, globals=globals if (globals is not None or not eval_str) else getattr(f, '__dict__', None), locals=locals)
@dispatch
def get_annotations(p: (functools.partial, functools.cached_property), **kwargs): return get_annotations(p.func, **kwargs)
@dispatch
def get_annotations(f: callable, *, eval_str=True, globals=None, locals=None): return get_annotations(_inspect_get_annotations(f), eval_str=eval_str, globals=globals if (globals is not None or not eval_str) else getattr(f, '__globals__', None), locals=locals)
@dispatch
def get_annotations(p: property, **kwargs): return get_annotations(p.fget, **kwargs)

def cast(*types): return lambda x: (t(i) if (not isinstance(i, t)) else i for t, i in zip(types, x))

@suppress_tb
def cast_call(f, *args, **kwargs):
	(f.__func__ if (isinstance(f, method)) else f).__annotations__ = get_annotations(f)
	fsig = _inspect_signature(f, eval_str=True)
	try:
		args = [(v.annotation)(args[ii]) if (v.annotation is not inspect._empty and not isinstance(args[ii], v.annotation)) else args[ii] for ii, (k, v) in enumerate(fsig.parameters.items()) if ii < len(args)]
		kwargs = {k: (fsig.parameters[k].annotation)(v) if (k in fsig.parameters and fsig.parameters[k].annotation is not inspect._empty and not isinstance(v, fsig.parameters[k].annotation)) else v for k, v in kwargs.items()}
	except Exception as ex: raise CastError() from ex
	r = f(*args, **kwargs)
	return ((fsig.return_annotation)(r) if (fsig.return_annotation is not inspect._empty) else r)
class CastError(TypeError): pass

@funcdecorator
def autocast(f):
	""" Beware! leads to undefined behavior when used with `@dispatch'. """

	return lambda *args, **kwargs: cast_call(f, *args, **kwargs)

@funcdecorator # XXX?
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

	#fsig = inspect.signature(f)
	#return lambda *args, **kwargs: f(*args, **S(kwargs) & {k: v.annotation() for k, v in fsig.parameters.items() if (k not in kwargs and v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and v.annotation is not inspect._empty)})

	return lambda *args, **kwargs: f(*args, **S(kwargs) & {k: v() for k, v in get_annotations(f).items() if k not in kwargs and not isinstance(v, str)})

def each(it): return tuple(it)

@dispatch
def tohashable(i: typing.Iterator): raise ValueError("Iterators are not hashable.")
@dispatch
def tohashable(d: typing.Dict): return tuple((k, tohashable(v)) for k, v in d.items())
@dispatch
def tohashable(s: str): return s
@dispatch
def tohashable(l: typing.Iterable): return tuple(map(tohashable, l))
@dispatch
def tohashable(x: typing.Hashable): return x

class cachedfunction:
	__slots__ = ('__wrapped__', '_cached', '_obj')

	class _noncached:
		__slots__ = ('value',)

		def __init__(self, value):
			self.value = value

	def __init__(self, f):
		self.__wrapped__ = f
		self._cached = dict()
		self._obj = None

	def __get__(self, obj, objcls):
		self._obj = obj
		return self

	@suppress_tb
	def __call__(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		k = tohashable((args, kwargs))
		try: return self._cached[k]
		except KeyError:
			try: r = self.__wrapped__(*args, **kwargs)
			except Exception as ex:
				ex.__context__ = None
				raise
			if (not isinstance(r, self._noncached)):
				self._cached[k] = r
				return r
			else: return r.value

	def nocache(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		return self.__wrapped__(*args, **kwargs)

	def is_cached(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		k = tohashable((args, kwargs))
		return (k in self._cached)

	def clear_cache(self):
		self._cached.clear()
class cachedclass(cachedfunction):
	def __subclasscheck__(self, subcls):
		return issubclass(subcls, self.__wrapped__)

	def __instancecheck__(self, obj):
		return isinstance(obj, self.__wrapped__)

@dispatch
def lrucachedfunction(f: callable): return lrucachedfunction()(f)
@dispatch
def lrucachedfunction(maxsize=None, typed=True):
	def decorator(f):
		f = functools.lru_cache(maxsize=maxsize, typed=typed)(f)
		f.nocache = f.__wrapped__
		f.clear_cache = f.cache_clear
		return f
	return decorator
def lrucachedclass(c): return lrucachedfunction(c)

class cachedclassproperty:
	class _empty: __slots__ = ()
	_empty = _empty()

	__slots__ = ('__wrapped__', '_cached')

	def __init__(self, f):
		self.__wrapped__ = f
		self._cached = self._empty

	def __get__(self, obj, objcls):
		if (self._cached is self._empty): self._cached = self.__wrapped__(obj)
		return self._cached

	def clear_cache(self):
		self._cached.clear()
try: cachedproperty = functools.cached_property
except AttributeError: cachedproperty = cachedclassproperty # TODO FIXME!
Scachedproperty = cachedclassproperty # TODO FIXME!

@dispatch
def allsubclasses(cls: type):
	""" Get all subclasses of class `cls' and its subclasses and so on recursively. """
	return (cls.__subclasses__() + [j for i in cls.__subclasses__() for j in allsubclasses(i)])
@dispatch
def allsubclasses(obj): return allsubclasses(type(obj))

@dispatch
def subclassdict(cls: type):
	""" Get name-class mapping for subclasses of class `cls'. """
	return {i.__name__: i for i in cls.__subclasses__()}
@dispatch
def subclassdict(obj): return subclassdict(type(obj))

@dispatch
def allsubclassdict(cls: type):
	""" Get name-class mapping for all subclasses of class `cls' (see `allsubclasses'). """
	return {i.__name__: i for i in allsubclasses(cls)}
@dispatch
def allsubclassdict(obj): return allsubclassdict(type(obj))

@dispatch
def allannotations(cls: type):
	""" Get annotations dict for all the MRO of class `cls' in right («super-to-sub») order. """
	return {k: v for i in cls.mro()[::-1] for k, v in get_annotations(i).items()}
@dispatch
def allannotations(obj): return allannotations(type(obj))

@dispatch
def allslots(cls: type):
	""" Get slots tuple for all the MRO of class `cls' in right («super-to-sub») order. """
	return tuple(j for i in cls.mro()[::-1] if hasattr(i, '__slots__') for j in i.__slots__)
@dispatch
def allslots(obj): return allslots(type(obj))

def spreadargs(f, okwargs, *args, **akwargs):
	fsig = _inspect_signature(f, eval_str=True)
	kwargs = (S(okwargs) & akwargs)
	try: kwnames = tuple(i[0] for i in fsig.parameters.items() if (assert_(i[1].kind != inspect.Parameter.VAR_KEYWORD) and i[1].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)))
	except AssertionError: kwnames = kwargs.keys()
	for i in kwnames:
		try: del okwargs[i]
		except KeyError: pass
	return f(*args, **kwargs)

def init(*names, **kwnames):
	@funcdecorator(signature=inspect.Signature((*(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names), *(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default) for name, default in kwnames.items()))))
	def decorator(f):
		def decorated(self, *args, **kwargs):
			missing = list()
			for i in names:
				try: setattr(self, i, kwargs.pop(i))
				except KeyError: missing.append(i)
			for k, v in kwnames.items():
				try: setattr(self, k, kwargs.pop(k))
				except KeyError:
					if (v is not ...): setattr(self, k, v() if (isinstance(v, (type, function, method))) else v)
			if (missing): raise TypeError(f"""{f.__name__}() missing {decline(len(missing), 'argument', 'arguments', sep=' required keyword-only ')}: {S(', ').join((i.join("''") for i in missing), last=(' and ', ', and '))}""")
			return f(self, *args, **kwargs)
		return decorated
	return decorator

@dispatch
def with_signature(fsig: str):
	def decorator(f):
		f.__text_signature__ = fsig
		inspect.signature(f)  # validate
		return f
	return decorator

logcolor = ('\033[94m', '\033[92m', '\033[93m', '\033[91m', '\033[95m')
noesc = re.compile(r'\033 (?: \] \w* ; \w* ; [^\033]* (?: \033\\ | \x07) | \[? [0-?]* [ -/]* [@-~]?)', re.X)
logfile = None
logoutput = sys.stderr
def log(l=None, *x, sep=' ', end='\n', fileend=None, ll=None, raw=False, tm=None, format=False, width=80, unlock=False, flush=True, nolog=False, nofile=False): # TODO: finally rewrite me as class pls
	""" Log anything. Print (formatted with datetime) to stderr and logfile (if set). Should be compatible with `builtins.print()'.
	Parameters:
		l?: level, must be >= global loglevel to print to stderr rather than only to logfile (or `/dev/null').
		*x: print-like args, what to log.
		sep: print-like separator.
		end: print-like line end.
		fileend: line end override for logfile.
		ll: formatted string instead of loglevel for output.
		raw: if true, do not print datetime/loglevel prefix if set.
		tm: specify datetime; current time if not set, nothing if false.
		format: if true, apply `pprint.pformat()' to args.
		width: specify output line width for wrapping; autodetect from stderr if not specified.
		unlock: if true, release all previously holded («locked») log messages.
		flush: if true, flush logfile if written to it.
		nolog: if true, force suppress printing to stderr.
		nofile: if true, force suppress printing to logfile.
	"""

	if (isinstance(log._logged_utils_start, tuple)):
		log._logged_utils_start, _logstateargs = True, log._logged_utils_start
		logstart('Utils')
		logstate(*_logstateargs)

	_l, _x = l, x
	if (l is None): l = ''
	if (type(l) is not int): l, x = None, (l, *x)
	if (x == ()): l, x = 0, (l,)

	x = (('plog():\n'*bool(format and not raw)) + sep.join(map(((lambda x: pprint.pformat(x, width=width, sort_dicts=False)) if (format) else str), x)))
	clearstr = Sstr(x).noesc()

	if (tm is None): tm = datetime.datetime.now().astimezone()

	if (not unlock and not log._loglock.empty()):
		log._loglock.put((copy.deepcopy((_l, *_x)), dict(sep=sep, end=end, fileend=fileend, raw=raw, tm=tm, nolog=nolog, nofile=nofile)))
		return clearstr

	if (not raw):
		fc = '\033[96m'
		try: lc = logcolor[l]
		except (TypeError, IndexError): lc = ''
		if (isinstance(tm, int)): tm = time.gmtime(tm)
		if (isinstance(tm, time.struct_time)): tm = time.strftime('[%x %X]', tm)
		elif (isinstance(tm, datetime.datetime)): tm = tm.strftime('[%x %X]')
		elif (isinstance(tm, datetime.date)): tm = tm.strftime('[%x]')
		elif (isinstance(tm, datetime.time)): tm = tm.strftime('[%X]')
		if (tm): tm = f"{fc}{tm}\033[m"
		if (ll is None): ll = (f"{fc}[\033[1m{lc}LV{l}{fc.replace('[', '[0;')}]\033[m" if (l is not None) else '')
		logstr = f"\033[K{tm}{' '*bool(tm)}{ll}{' '*bool(ll)}\033[1m{x}\033[m"
	else: logstr = str(x)

	if (unlock and not log._loglock.empty()):
		ul = list()
		for i in iter_queue(log._loglock):
			if (i is None): break
			ul.append(i)
		for i in ul[::-1]:
			log(*i[0], **i[1])

	if (fileend is None): fileend = end
	if (logfile and not nofile): print(logstr, end=fileend, file=logfile, flush=flush)
	if (not nolog and (l or 0) <= loglevel): print(logstr, end=end, file=logoutput, flush=True)

	return clearstr
log._loglock = queue.LifoQueue()
log._logged_start = dict()
log._logged_utils_start = None
def plog(*args, **kwargs): parseargs(kwargs, format=True); return log(*args, **kwargs)
def _dlog(*args, prefix='', **kwargs): parseargs(kwargs, ll=f"\033[95m[\033[1mDEBUG\033[0;95m]\033[0;96m{prefix}", tm=''); return log(*args, **kwargs)
def dlog(*args, **kwargs): return _dlog(*map(str, args), **kwargs)
def dplog(*args, **kwargs): parseargs(kwargs, format=True, sep='\n'); return _dlog(*args, **kwargs)
def rlog(*args, **kwargs): parseargs(kwargs, raw=True); return log(*args, **kwargs)
def logdumb(**kwargs): return log(raw=True, end='', **kwargs)
def logstart(x):
	""" from utils[.nolog] import *; logstart(name) """
	if (log._logged_utils_start is None): log._logged_utils_start = False; return
	#if ((name := inspect.stack(0)[1].frame.f_globals.get('__name__')) is not None):
	if (log._logged_start.get(x) is True): return
	log._logged_start[x] = True
	log(x+'\033[m...', end=' ') #, nolog=(x == 'Utils'))
	locklog()
def logstate(state, x=''):
	if (log._logged_utils_start is False): log._logged_utils_start = (state, x); return
	log(state+(': '+str(x))*bool(str(x))+'\033[m', raw=True, unlock=True)
def logstarted(x=''): """ if (__name__ == '__main__'): logstart(); main() """; logstate('\033[94mstarted', x)
def logimported(x=''): """ if (__name__ != '__main__'): logimported() """; logstate('\033[96mimported', x)
def logok(x=''): logstate('\033[92mok', x)
def logex(x=''): logstate('\033[91merror', unraise(x))
def logwarn(x=''): logstate('\033[93mwarning', x)
def setlogoutput(f): global logoutput; logoutput = f
def setlogfile(f): global logfile; logfile = open(f, 'a') if (isinstance(f, str)) else f
def setloglevel(l): global loglevel; loglevel = l
def locklog(): log._loglock.put(None)
def unlocklog(): logdumb(unlock=True)
def logflush(): logdumb(flush=True)
def setutilsnologimport(): log._logged_utils_start = True

_logged_exceptions = set()
@dispatch
def exception(ex: BaseException, extra=None, *, once=False, raw=False, nolog=False, nohandlers=False, _dlog=False, _caught=False, **kwargs):
	""" Log an exception.
	Parameters:
		ex: exception to log.
		extra: additional info to log.
		nolog: do not write to stderr.
		nohandlers: do not call exc_handlers.
	"""
	ex = unraise(ex)
	if (isinstance(ex, NonLoggingException)): return
	if (once):
		if (repr(ex) in _logged_exceptions): return
		_logged_exceptions.add(repr(ex))
	e = (dlog if (_dlog) else log)(('\033[91m'+'Caught '*_caught if (not isinstance(ex, Warning)) else '\033[93m' if ('warning' in ex.__class__.__name__.casefold()) else '\033[91m')+ex.__class__.__qualname__+(' on line '+' -> '.join(terminal_link((f"file://{socket.gethostname()}" + os.path.realpath(i[0].f_code.co_filename)) if (os.path.exists(i[0].f_code.co_filename)) else i[0].f_code.co_filename, i[1]) for i in traceback.walk_tb(ex.__traceback__) if i[1])).removesuffix(' on line')+'\033[m'+(': '+str(ex))*bool(str(ex))+('\033[0;2m; ('+str(extra)+')\033[m' if (extra is not None) else ''), raw=raw, nolog=nolog, **kwargs)
	if (not nohandlers):
		for i in log._exc_handlers:
			try: i(e, ex)
			except Exception: pass
log._exc_handlers = set()
def register_exc_handler(f): log._exc_handlers.add(f)
def logexception(*args, **kwargs): return exception(*args, **kwargs, _caught=True)
def dlogexception(*args, **kwargs): return logexception(*args, **kwargs, _dlog=True)

def dcall(f): logexception(DeprecationWarning(" *** dcall() → tracecall() *** ")); return tracecall(f)

@funcdecorator
def tracecall(f):
	def decorated(*args, **kwargs):
		fcall = f"{f.__name__}({', '.join((*map(repr, args), *(f'{k}={repr(v)}' for k, v in kwargs.items())))})"
		dlog(f"→ {fcall}", end='\n\n')
		try: r = f(*args, **kwargs)
		except Exception as ex: dlogexception(ex, prefix=f"\033[39m × {fcall}\n", end='\n\n'); raise
		else: dlog(f"← {fcall} -> {repr(r)}", end='\n\n'); return r
	return decorated

@funcdecorator
def atracecall(f):
	async def decorated(*args, **kwargs):
		fcall = f"{f.__name__}({', '.join((*map(repr, args), *(f'{k}={repr(v)}' for k, v in kwargs.items())))})"
		dlog(f"→ await {fcall}", end='\n\n')
		try: r = await f(*args, **kwargs)
		except Exception as ex: dlogexception(ex, prefix=f"\033[39m × {fcall}\n", end='\n\n'); raise
		else: dlog(f"← await {fcall} -> {repr(r)}", end='\n\n'); return r
	return decorated

@suppress_tb
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

class DB:
	""" All-in-one lightweight database class. """

	class _empty: __slots__ = ()

	__slots__ = ('file', 'fields', 'serializer', 'lock', 'backup', 'sensitive', 'nolog')

	def __init__(self, file=None, serializer=pickle):
		self.lock = threading.Lock()
		self.setfile(file)
		self.setserializer(serializer)
		self.fields = dict()
		self.nolog = False
		self.backup = True
		self.sensitive = False

	@dispatch
	def setfile(self, file: NoneType):
		with self.lock:
			self.file = None

		return False

	@dispatch
	def setfile(self, file: str):
		if ('/' not in file): file = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), file)
		file = os.path.expanduser(file)

		with self.lock:
			try: self.file = open(file, 'r+b')
			except FileNotFoundError: self.file = open(file, 'w+b')
			if (self.sensitive): os.fchmod(self.file.fileno(), 0o600)

		return True

	def setnolog(self, nolog=True):
		with self.lock:
			self.nolog = bool(nolog)

	def setbackup(self, backup):
		with self.lock:
			self.backup = bool(backup)

	def setsensitive(self, sensitive=True):
		with self.lock:
			if (not sensitive and self.sensitive and self.file is not None):
				umask = os.umask(0); os.umask(umask)
				os.fchmod(self.file.fileno(), 0o666 ^ umask)
			self.sensitive = bool(sensitive)

	@dispatch
	def setserializer(self, serializer: module):
		with self.lock:
			self.serializer = serializer

	def register(self, *fields):
		globals = inspect.stack(0)[1].frame.f_globals

		with self.lock:
			for field in fields:
				self.fields[field] = (globals, globals.get('__annotations__', {}).get(field, self._empty), globals.get(field, self._empty))

	def load(self, nolog=None):
		with self.lock:
			nolog = (self.nolog if (nolog is None) else nolog)

			if (not self.file): return
			if (not nolog): logstart("Loading database")

			db = dict()

			try: db = self.serializer.load(self.file)
			except EOFError:
				if (not nolog): logwarn("database is empty")
			else:
				if (not nolog): logok()

			self.file.seek(0)

			for field, (globals, annotation, default) in self.fields.items():
				try: value = db[field]
				except KeyError:
					if (not nolog): log(1, f"Not in DB: {field}")
					if (field not in globals):
						try: globals[field] = (default if (default is not self._empty) else annotation())
						except TypeError: pass
				else: globals[field] = value

		return db

	def save(self, db=None, backup=None, nolog=None):
		with self.lock:
			nolog = (self.nolog if (nolog is None) else nolog)
			backup = (self.backup if (backup is None) else backup)

			if (not self.file): return
			if (not nolog): logstart("Saving database")

			if (backup):
				try: os.mkdir('backup')
				except FileExistsError: pass
				try: shutil.copyfile(self.file.name, f"backup/{self.file.name if (hasattr(self.file, 'name')) else ''}_{int(time.time())}.db")
				except OSError: pass

			try:
				if (db is None): db = {field: globals[0][field] if (isinstance(globals, tuple)) else globals[field]
				                       for field, globals in self.fields.items()
				                       if (not isinstance(globals, tuple) and field in globals)
				                          or (isinstance(globals, tuple) and field in globals[0]
				                              and globals[2] is not None and globals[0][field] != globals[2])}
				self.serializer.dump(db, self.file)
			except Exception as ex:
				if (not nolog): logex(ex)
			else:
				if (not nolog): logok()

			self.file.truncate()
			self.file.seek(0)
db = DB()

def progress(cv, mv, *, pv="▏▎▍▌▋▊▉█", fill='░', border='│', prefix='', fixed=True, print=True, **kwargs): # TODO: optimize
	return getattr(Progress(mv, chars=pv, border=border, prefix=prefix, fixed=fixed, **kwargs), 'print' if (print) else 'format')(cv)

class Progress:
	__slots__ = ('mv', 'chars', 'border', 'fill', 'prefix', 'fixed', 'add_base', 'add_speed_eta', 'printed', 'started', '_pool')

	def __init__(self, mv=None, *, chars=' ▏▎▍▌▋▊▉█', border='│', fill=' ', prefix='', fixed=False, add_base=False, add_speed_eta=False, _pool=None):
		if (fixed and mv is None): raise ValueError("`mv' must be specified when `fixed=True`")
		self.mv, self.chars, self.border, self.fill, self.prefix, self.fixed, self.add_base, self.add_speed_eta, self._pool = mv, chars, border, fill, prefix, fixed, add_base, add_speed_eta, _pool
		self.printed = bool()
		self.started = None

	def __enter__(self):
		self.started = time.time()
		return self

	def __exit__(self, exc_type, exc, tb):
		if (self.printed and self._pool is None and sys.stderr.isatty()): print(file=sys.stderr, flush=True)

	def format(self, cv, width, *, add_base=None, add_speed_eta=None):
		if (add_base is None): add_base = self.add_base
		if (add_speed_eta is None): add_speed_eta = self.add_speed_eta
		if (self.started is None or not cv): add_speed_eta = False

		l = (self.l(add_base) if (self._pool is None) else max(i.l(add_base) for i in self._pool.p))

		fstr = (self.prefix + ('%s/%s (%d%%%s) ' if (not self.fixed) else f"%{l}s/%-{l}s (%-3d%%%s) "))
		r = (fstr % (*((cv, self.mv) if (not add_base) else ((self.format_base(i, base=add_base)) for i in (cv, self.mv))), (cv*100//self.mv if (self.mv) else 0), ', '+self.format_speed_eta(cv, self.mv, (time.time() - self.started), fixed=self.fixed, add_base=add_base) if (add_speed_eta) else ''))

		w = len(r)
		if (self._pool is not None):
			if (w > self._pool.mw): self._pool.mw = w
			else: w = self._pool.mw

		return (r.rjust(w) + self.format_bar(cv, self.mv, (width - w), chars=self.chars, border=self.border, fill=self.fill))

	@staticmethod
	def format_bar(cv, mv, width, *, chars=' ▏▎▍▌▋▊▉█', border='│', fill=' '):
		cv = max(0, min(mv, cv))
		if (not isiterablenostr(border)): border = (border, border)
		d = 100/(width-2)
		fp, pp = (divmod(cv*100/d, mv) if (mv) else (0, 0))
		pb = (chars[-1]*int(fp) + (chars[int(pp*len(chars)//mv)]*(cv != mv) if (mv) else ''))
		return f"{border[0]}{pb.ljust(width-2, fill)}{border[1]}"

	@classmethod
	def format_speed_eta(cls, cv, mv, elapsed, *, fixed=False, add_base=None):
		speed = cv/elapsed
		eta = (math.ceil(mv/speed)-elapsed if (speed) else 0)
		if (fixed): return ((first(f"%2d{'dhms'[ii]}" % i for ii, i in enumerate((eta//60//60//24, eta//60//60%24, eta//60%60, eta%60)) if i) if (eta > 0) else ' ? ') + ' ETA')
		eta = (' '.join(f"{int(i)}{'dhms'[ii]}" for ii, i in enumerate((eta//60//60//24, eta//60//60%24, eta//60%60, eta%60)) if i) if (eta > 0) else '?')
		speed_u = 0
		for i in (60, 60, 24):
			if (speed < 1): speed *= i; speed_u += 1
		if (add_base is not None): speed = cls.format_base(speed, base=add_base)
		return f"{speed}/{'smhd'[speed_u]}, {eta} ETA"

	@staticmethod
	def format_base(cv, base=True, *, _calc_len=False):
		""" base: `(step, names)' [ex: `(1024, ('b', 'kb', 'mb', 'gb', 'tb')'], default: `(1000, ' KMB')'
		names should be non-decreasing in length for correct use in fixed-width mode.
		whitespace character will be treated as nothing.
		"""

		if (base is True): base = (1000, ' KMB')
		step, names = base

		l = len(names)
		if (_calc_len): sl, nl = len(S(step))-1, max(map(len, names))

		try: return first((f"{{: {sl+nl}}}".format(v) if (_calc_len) else str(v))+(b if (b != ' ') else '') for b, v in ((i, cv//(step**(l-ii))) for ii, i in enumerate(reversed(names), 1)) if v) # TODO: fractions; negative
		except StopIteration: return '0'

	def print(self, cv, *, out=sys.stderr, width=None, flush=True):
		if (width is None and out is sys.stderr):
			try: width = os.get_terminal_size(out.fileno())[0]
			except OSError: pass

		if (width is None): return

		print('\033[K' + self.format(cv, width=width), end='\r', file=out, flush=flush)

		self.printed = (out is sys.stderr)

	def l(self, add_base):
		return (len(S(self.mv)) if (not add_base) else len(self.format_base(self.mv, base=add_base, _calc_len=True)))

class ProgressPool:
	#@dispatch
	#def __init__(self, *p: Progress, **kwargs):
	#	self.p, self.kwargs = list(p), parseargs(kwargs, fixed=True)
	#	for i in self.p:
	#		i._pool = self
	#	self.ranges = list()

	@dispatch
	def __init__(self, n: int = 0, **kwargs):
		#self.__init__(*(Progress(-1, **kwargs, _pool=self) for _ in range(n)), **kwargs)
		self.kwargs = kwargs
		self.p = [Progress(-1, **parseargs(kwargs, fixed=True), _pool=self) for _ in range(n)]
		self.mw = int()
		self.ranges = list()
		self.printed = bool()

	def __enter__(self):
		self.started = time.time()
		return self

	def __exit__(self, exc_type, exc, tb):
		if (self.printed and sys.stderr.isatty()): print(end='\033[J', file=sys.stderr, flush=True)

	def print(self, *cvs, width=None):
		if (not sys.stderr.isatty()): return

		ii = None
		for ii, (p, cv) in enumerate(zip(self.p, cvs)):
			if (ii): print(file=sys.stderr, flush=True)
			p.print(cv, width=width)
		if (ii): print(end=f"\033[{ii}A", file=sys.stderr, flush=True)

		self.printed = True

	def range(self, start, stop=None, step=1, **kwargs):
		if (stop is None): start, stop = 0, start

		n = len(self.ranges)
		self.ranges.append(int())

		cv = (stop - start)
		if (n == len(self.p)): self.p.append(Progress(cv, **parseargs(kwargs, **self.kwargs), _pool=self))
		else: self.p[n].mv = cv

		for i in range(start, stop, step):
			self.ranges[n] = (i - start)
			if (n == len(self.p)-1): self.print(*self.ranges)
			yield i

		self.ranges[n] = stop
		self.print(*self.ranges)
		self.ranges.pop()

		self.p.pop()

	@dispatch
	def iter(self, iterator: typing.Iterator, l: int, step: int = 1, **kwargs):
		yield from (next(iterator) for _ in self.range(l, step=step, **kwargs))

	@dispatch
	def iter(self, iterable: typing.Iterable, **kwargs):
		it = tuple(iterable)
		yield from (it[i] for i in self.range(len(it), **kwargs))

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
		return super().__enter__()

	def __exit__(self, exc_type, exc, tb):
		self.stop()
		super().__exit__(exc_type, exc, tb)

	def run(self):
		while (not self.stopped):
			locklog()
			self.print(*self.cvs, width=self.width)
			unlocklog()
			time.sleep(self.delay)

	def range(self, start, stop=None, step=1, **kwargs):
		if (stop is None): start, stop = 0, start
		n = self.ranges
		self.ranges += 1

		cv = (stop - start)
		if (n == len(self.p)):
			self.p.append(Progress(cv, **parseargs(kwargs, **self.kwargs), _pool=self))
			self.cvs.append(int())
		else: self.p[n].mv = cv

		for i in range(start, stop, step):
			self.cvs[n] = i-start
			yield i

		self.ranges -= 1
		self.p.pop()

	def stop(self):
		self.stopped = True
		self.join()
		if (sys.stderr.isatty()): print(end='\r\033[K', file=sys.stderr, flush=True)

def progrange(start, stop=None, step=1, **kwargs):
	parseargs(kwargs, fixed=True)
	if (stop is None): start, stop = 0, start

	cv = (stop - start)
	with Progress(cv, **kwargs) as p:
		for i in range(start, stop, step):
			p.print(i - start)
			yield i
		p.print(stop)

@dispatch
def progiter(iterator: typing.Iterator, l: int): # TODO: why yield?
	return (next(iterator) for _ in progrange(l))

@dispatch
def progiter(iterable: typing.Iterable):
	l = tuple(iterable)
	return (i for _, i in zip(progrange(len(l)), l))

def testprogress(n=1000, sleep=0.002, **kwargs):
	parseargs(kwargs, fixed=True)
	with Progress(n, **kwargs) as p:
		for i in range(n+1):
			p.print(i)
			time.sleep(sleep)

class NodesTree:
	chars = '┌├└─│╾╼'
	colorchars = tuple(f"\033[2m{i}\033[22m" for i in chars)

	def __init__(self, x):
		self.tree = x

	def print(self, out=sys.stdout, **fmtkwargs):
		parseargs(fmtkwargs, color=out.isatty())
		print(self.format(**fmtkwargs), file=out, flush=True)

	def format(self, **fmtkwargs):
		parseargs(fmtkwargs, root=True)
		return '\n'.join(self.format_node(self.tree, **fmtkwargs))

	@classmethod
	def format_node(cls, node, indent=2, color=False, usenodechars=False, *, root=False):
		chars = cls.colorchars if (color) else cls.chars
		nodechar = chars[6] if (usenodechars) else chars[3]
		for ii, (k, v) in enumerate(node.items()):
			if (isiterable(v) and not isinstance(v, (str, dict))): k, v = v
			yield ((chars[0] if (len(node) > 1) else chars[5]) if (root and ii == 0) else chars[1] if (ii < len(node)-1) else chars[2])+chars[3]*(indent-1)+nodechar+' '+str(k)
			it = tuple(cls.format_node(v if (isinstance(v, dict)) else {v: {}}, indent=indent, color=color, usenodechars=usenodechars))
			yield from map(lambda x: (chars[4] if (ii < len(node)-1) else ' ')+' '*(indent+1)+x, it)

class TaskTree(NodesTree):
	class Task:
		__slots__ = ('title', 'state', 'subtasks', 'printed')

		def __init__(self, title, subtasks=None):
			self.title, self.subtasks = str(title), (subtasks if (subtasks is not None) else [])
			self.state = None

	def __init__(self, x, **kwargs):
		self.tree = x
		self.l = len(tuple(self.format_node(self.tree, root=True, **kwargs)))
		self.printed = bool()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		if (self.printed and sys.stderr.isatty()): print(end='\n'*self.l, file=sys.stderr, flush=True)

	def print(self, **fmtkwargs):
		if (not sys.stderr.isatty()): return

		print(end='\033[J', file=sys.stderr)
		print(*(x+' '+self.format_task(y) for x, y in self.format_node(self.tree, root=True, **fmtkwargs)), sep='\n', end=f"\r\033[{self.l-1}A", file=sys.stderr, flush=True)

		self.printed = True

	def format_node(self, node, indent=2, color=False, *, root=False):
		chars = self.colorchars if (color) else self.chars
		for ii, t in enumerate(node):
			yield (((chars[0] if (len(node) > 1) else chars[5]) if (root and ii == 0) else chars[1] if (ii < len(node)-1) else chars[2])+chars[3]*(indent-1)+(chars[6] if (t.state) else chars[3]), t)
			it = tuple(self.format_node(t.subtasks, indent=indent, color=color))
			yield from map(lambda x: ((chars[4] if (ii < len(node)-1) else ' ')+' '*indent+x[0], x[1]), it)

	@staticmethod
	def format_task(t):
		return f"\033[{93 if (t.state is None) else 91 if (not t.state) else 92}m{t.title}\033[m"

def validate(l, d, nolog=False):
	for i in d:
		try:
			t, e = d[i] if (type(d[i]) == tuple) else (d[i], 'True')
			r = eval(e.format(t(l[i]))) if (type(e) == str) else e(t(l[i]))
			if (bool(r) == False): raise ValidationError(r)
		except Exception as ex:
			if (not nolog): log(2, "\033[91mValidation error:\033[m %s" % ex)
			return False
	return True
class ValidationError(AssertionError): pass

#@with_signature("(n, first[, second[, third[, fourth]], other], /, *, zeroth=other, prefix='', sep=' ', format=False, show_one=True)")
def decline(n, first, second=None, third=None, fourth=None, other=None, /, *, zeroth=None, prefix='', sep=' ', format=False, show_one=True):
	""" decline(n, first[, second[, third[, fourth]], other], /, *, zeroth=other, prefix='', sep=' ', format=False, show_one=True) """

	if (not isinstance(first, str)):
		l = list(first)
		first = l.pop(0)
		if (l): second = l.pop(0)
		if (l): third = l.pop(0)
		if (l): fourth = l.pop(0)
		if (l): other = l.pop(0)
		if (l): raise ValueError(l)
	if (second is None): second = first
	if (third is None): third = second
	if (fourth is None): third, fourth, other = second, second, third
	if (other is None): other = fourth
	if (zeroth is None): zeroth = other

	if (isinstance(prefix, str)): prefix = (prefix,)*5
	elif (len(prefix) == 1): prefix *= 5
	elif (len(prefix) == 2): prefix = (*prefix, prefix[-1])

	if (5 <= abs(n % 100) <= 20): q = 9
	else: q = abs(n) % 10
	if (q == 0): r, p = zeroth, prefix[-1]
	elif (q == 1): r, p = first, prefix[0]
	elif (q == 2): r, p = second, prefix[1]
	elif (q == 3): r, p = third, prefix[2]
	elif (q == 4): r, p = fourth, prefix[-2]
	else: r, p = other, prefix[-1]
	return f"{p}{str(S(n).format(' ' if (format is True) else format) if (format) else n if (format != '') else format)+sep if (n != 1 or show_one) else ''}{r}"
def testdecline(*args, **kwargs): return '\n'.join(decline(i, *args, **kwargs) for i in range(31))

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

class _raise: pass
@suppress_tb
def first(l, default=_raise):
	try: return next(iter(l))
	except StopIteration:
		if (default is _raise): raise
		return default
@suppress_tb
def last(l):
	l = iter(l)
	r = next(l)
	while (True):
		try: r = next(l)
		except StopIteration: return r
@suppress_tb
def only(l):
	l = iter(l)
	try: return next(l)
	finally:
		try: next(l)
		except StopIteration: pass
		else: raise StopIteration("Only a single value expected")

def pm(x): return (+1 if (x) else -1)
def constrain(x, lb, ub): return min(ub, max(lb, x))
def prod(x, initial=1): return functools.reduce(operator.mul, x, initial)
def average(x, default=None): return sum(x)/len(x) if (x) else default if (default is not None) else raise_(ValueError("average() arg is an empty sequence"))

global_lambdas = list() # TODO: WeakSet
def global_lambda(l): global_lambdas.append(l); return l

@dispatch
@suppress_tb
def singleton(C: type): C.__new__ = cachedfunction(C.__new__); return C()
@dispatch
@suppress_tb
def singleton(*args, **kwargs): return lambda C: C(*args, **kwargs)

class SingletonMeta(type):
	_ready = bool()

	@cachedfunction
	@suppress_tb
	def __new__(metacls, name, bases, classdict):
		r = super().__new__(metacls, name, bases, classdict)
		if (metacls._ready): return r()
		metacls._ready = True
		return r
class Singleton(metaclass=SingletonMeta):
	def __repr__(self):
		return f"<Singleton {self.__class__.__qualname__}>"

@singleton
class clear:
	""" Clear the terminal. """

	def __call__(self):
		print(end='\033c', file=sys.stderr, flush=True)

	def __repr__(self):
		if (not __repl__ and sys.exc_info()[0] is None): return super().__repr__()
		self()
		return ''

@singleton
class termsize:
	def __str__(self):
		return f"{self.width}×{self.height}"

	@property
	def columns(self):
		return os.get_terminal_size().columns
	width = columns

	@property
	def lines(self):
		return os.get_terminal_size().lines
	height = lines

class lc:
	__slots__ = ('category', 'lc', 'pl')

	@dispatch
	def __init__(self):
		self.__init__(locale.LC_ALL)

	@dispatch
	def __init__(self, lc: (str, typing.Iterable, NoneType)):
		self.__init__(locale.LC_ALL, lc)

	@dispatch
	def __init__(self, category: int = locale.LC_ALL, lc: (str, typing.Iterable, NoneType) = None):
		self.category, self.lc = category, (lc if (lc is not None) else '.'.join(locale.getlocale()))

	def __enter__(self):
		self.pl = locale.setlocale(self.category)
		locale.setlocale(self.category, self.lc)

	def __exit__(self, type, value, tb):
		locale.setlocale(self.category, self.pl)

class ll:
	def __init__(self, ll):
		self.ll = ll

	def __enter__(self):
		self.pl = loglevel
		setloglevel(self.ll)

	def __exit__(self, type, value, tb):
		setloglevel(self.pl)

class timecounter(Singleton):
	def __init__(self):
		self.started = None
		self.ended = None

	def __call__(self):
		return self

	def __enter__(self):
		if (self is timecounter): self = type(self)()
		self.started = time.perf_counter()
		return self

	def __exit__(self, type, value, tb):
		self.ended = time.perf_counter()

	def time(self):
		if (self.started is None): return 0
		if (self.ended is None): return (time.perf_counter() - self.started)
		return (self.ended - self.started)

class classonlymethod:
	__slots__ = ('__wrapped__',)

	def __init__(self, f):
		self.__wrapped__ = f

	@suppress_tb
	def __get__(self, obj, cls=None):
		if (obj is not None): raise AttributeError(f"'{cls.__name__}.{self.__wrapped__.__name__}()' is a class-only method.")
		return functools.partial(self.__wrapped__, cls)

class instancemethod:
	__slots__ = ('__wrapped__',)

	def __init__(self, f):
		self.__wrapped__ = f

	def __get__(self, obj, cls):
		return functools.partial(self.__wrapped__, (obj or cls))

class classproperty:
	__slots__ = ('__wrapped__',)

	def __init__(self, f):
		self.__wrapped__ = f

	def __get__(self, obj, cls):
		return self.__wrapped__(cls)

class instanceproperty:
	__slots__ = ('__wrapped__',)

	def __init__(self, f):
		self.__wrapped__ = f

	def __get__(self, obj, cls):
		return self.__wrapped__(obj or cls)

class attrget:
	__slots__ = ('__wrapped__',)

	class getter:
		__slots__ = ('obj', '__wrapped__')

		def __init__(self, obj, f):
			self.obj, self.__wrapped__ = obj, f

		def __getattr__(self, x):
			return self.__wrapped__(self.obj, x)

	def __init__(self, f):
		self.__wrapped__ = f

	def __get__(self, obj, cls):
		return self.getter(obj, self.__wrapped__)

class itemget:
	__slots__ = ('__wrapped__', '__call__', '_bool')

	class getter:
		__slots__ = ('obj', '__wrapped__')

		def __init__(self, obj, f):
			self.obj, self.__wrapped__ = obj, f

		def __contains__(self, x):
			try: self[x]
			except KeyError: return False
			else: return True

		def __getitem__(self, x):
			return self.__wrapped__(self.obj, *(x if (isinstance(x, tuple)) else (x,)))

	@dispatch
	def __init__(self, *, bool: function):
		self._bool = bool
		self.__call__ = lambda f: each[setattr(self, '__wrapped__', f), delattr(self, '__call__')]

	@dispatch
	def __init__(self, f):
		self.__wrapped__ = f

	def __bool__(self):
		try: f = self._bool
		except AttributeError: return super().__bool__()
		else: return f(self.obj)

	def __get__(self, obj, cls):
		return self.getter(obj, self.__wrapped__)

class classitemget(itemget):
	__slots__ = ('__wrapped__',)

	def __init__(self, f):
		self.__wrapped__ = f

	def __get__(self, obj, cls):
		return self.getter(cls, self.__wrapped__)

class staticitemget:
	__slots__ = ('__wrapped__', '_fkeys')

	def __init__(self, f):
		self.__wrapped__ = f
		self._fkeys = lambda self: ()

	def __iter__(self):
		raise ValueError("staticitemget is not iterable")

	def __getitem__(self, x):
		return (self.__wrapped__(*x) if (isinstance(x, tuple)) else self.__wrapped__(x))

	def __contains__(self, x):
		if (x in self.keys()): return True
		try: self[x]
		except KeyError: return False
		else: return True

	def __call__(self, *args, **kwargs):
		return self.__wrapped__(*args, **kwargs)

	def fkeys(self, f):
		self._fkeys = f
		return f

	def keys(self):
		return self._fkeys(self)
class staticitemgetclass(staticitemget):
	def __subclasscheck__(self, subcls):
		return issubclass(subcls, self.__wrapped__)

	def __instancecheck__(self, obj):
		return isinstance(obj, self.__wrapped__)

class staticattrget:
	__slots__ = ('__wrapped__',)

	def __init__(self, f):
		self.__wrapped__ = f

	def __getattr__(self, x):
		return self.__wrapped__(x)

	def __call__(self, *args, **kwargs):
		return self.__wrapped__(*args, **kwargs)

class AttrView:
	def __init__(self, obj):
		self.obj = obj

	def __repr__(self):
		return f"<{self.__class__.__name__} of {self.obj}>"

	def __contains__(self, k):
		if (not isinstance(k, str)): return False
		return hasattr(self.obj, k)

	def __getitem__(self, k):
		if (not isinstance(k, str)): raise KeyError(k)
		return getattr(self.obj, k)

	def __setitem__(self, k, v):
		if (not isinstance(k, str)): raise KeyError(k)
		setattr(self.obj, k, v)

	def keys(self):
		return dir(self.obj)

class AttrProxy:
	def __init__(self, *objects):
		self.objects = objects

	def __repr__(self):
		return f"<{self.__class__.__name__} of {S(', ').join(self.objects)}>"

	def __getitem__(self, x):
		for i in self.objects:
			try: return getattr(i, x)
			except AttributeError: continue
		else: raise KeyError(x)

	def __dir__(self):
		return [k for i in self.objects for k in dir(i)]

	def __iter__(self):
		return iter(self.keys())

	def __dict__(self):
		return {k: getattr(i, k) for i in self.objects for k in dir(i)}

	def keys(self):
		return self.__dir__()

class UserDict(collections.UserDict):
	__slots__ = ('_UserDict__dict',)

	def __init__(self, dict=None, /, **kwargs):
		super().__setattr__('_UserDict__dict', {})
		if (dict is not None): self.update(dict)
		if (kwargs): self.update(kwargs)

	def __len__(self):
		return len(self.__dict)

	def __getitem__(self, x):
		return self.__dict[x]

	def __setitem__(self, k, v):
		self.__dict[k] = v

	def __delitem__(self, x):
		del self.__dict[x]

	def __iter__(self):
		return iter(self.__dict)

	def __contains__(self, x):
		return (x in self.__dict)

	def __repr__(self):
		return repr(self.__dict)

	def __or__(self, other):
		if (isinstance(other, UserDict)): return self.__class__(self.__dict | other.__dict)
		elif (isinstance(other, dict)): return self.__class__(self.__dict | other)
		else: return NotImplemented

	def __ror__(self, other):
		if (isinstance(other, UserDict)): return self.__class__(other.__dict | self.__dict)
		elif (isinstance(other, dict)): return self.__class__(other | self.__dict)
		else: return NotImplemented

	def __ior__(self, other):
		if (isinstance(other, UserDict)): self.__dict.update(other.__dict)
		else: self.__dict.update(other)
		return self

	def copy(self):
		return self.__class__.of(self.__dict.copy())

	@property
	def __dict__(self):
		return self.__dict

	@classmethod
	def fromkeys(cls, iterable, value=None):
		d = cls()
		for k in iterable:
			d[k] = value
		return d

	@classonlymethod
	def of(cls, data: dict):
		d = cls()
		super().__setattr__(d, '_UserDict__dict', data)
		return d

class AttrDict(UserDict):
	__slots__ = ()

	def __getitem__(self, x):
		try: r = super().__getitem__(x)
		except KeyError:
			try: __missing__ = self.__class__.__missing__
			except AttributeError:
				raise KeyError(x)
			else: r = __missing__(self, x)

		if (isinstance(r, dict) and not isinstance(r, AttrDict)): return AttrDict.of(r)
		else: return r

	def __getattr__(self, x):
		try: return super().__getattribute__(x)
		except AttributeError as ex:
			try: return self[x]
			except KeyError as ex: e = ex

		e.__suppress_context__ = True
		raise AttributeError(*e.args) from e.with_traceback(None)

	def __setattr__(self, k, v):
		self[k] = v

	def __delattr__(self, x):
		del self[x]

class DefaultAttrDict(AttrDict):
	__slots__ = ('default_factory',)

	def __init__(self, default, *args, **kwargs):
		super().__init__(*args, **kwargs)
		object.__setattr__(self, 'default_factory', default)

	def __getitem__(self, x):
		try: return super().__getitem__(x)
		except KeyError: return self.__missing__()

	def __missing__(self, x):
		if (self.default_factory is None): raise KeyError(x)
		r = self[x] = self.default_factory()
		return r

class Builder:
	def __init__(self, cls, *args, **kwargs):
		self.cls = cls
		self.calls = collections.deque(((None, args, kwargs),))

	def __repr__(self):
		return f"<{self.__class__.__name__} of {repr(self.cls).removeprefix('<').removesuffix('>')}>"

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
			return f"<{self.__class__.__name__} '{self.name}'>"

	def __prepare__(name, bases):
		return type('', (dict,), {'__getitem__': lambda self, x: MetaBuilder.Var(x[2:]) if (x.startswith('a_') and x[2:]) else dict.__getitem__(self, x)})()

class SlotsOnlyMeta(type):
	def __new__(metacls, name, bases, classdict):
		annotations = get_annotations(classdict)
		classdict['__slots__'] = tuple(k for k, v in annotations.items() if not (isinstance(p := classdict.get(k), (property, functools.cached_property)) and (ra := get_annotations(p).get('return')) and ra == v))
		return super().__new__(metacls, name, bases, classdict)
class SlotsOnly(metaclass=SlotsOnlyMeta): pass
class ABCSlotsOnlyMeta(SlotsOnlyMeta, abc.ABCMeta): pass
class ABCSlotsOnly(metaclass=ABCSlotsOnlyMeta): pass

def typecheck(x, t): return dispatch._dispatch__typecheck(x, t)
#@dispatch
#def typecheck(x, t: lambda t: typing_inspect.get_origin(t) is type and typing_inspect.is_generic_type(t)): return issubclass(x, typing_inspect.get_args(t))
#@dispatch
#def typecheck(x, t: type): return isinstance(x, t)
#@dispatch
#def typecheck(x, t: typing_inspect.is_generic_type):

class SlotsTypecheckMeta(type):
	def __new__(metacls, name, bases, classdict):
		annotations = get_annotations(classdict)

		for k, v in classdict.items():
			if (not isinstance(v, (property, functools.cached_property))): continue
			try: ra = get_annotations(v)['return']
			except KeyError: continue
			if (k in annotations):
				if ((a := annotations[k]) is not ... and ra != a and not typecheck(ra, type[a])):
					raise TypeError(f"Specified property type annotation ({k} -> {format_inspect_annotation(ra)}) is not a subclass of its slot annotation ({k}: {format_inspect_annotation(a)})")
			else: annotations[k] = ra

		for k, (v, c) in functools.reduce(operator.or_, ({k: (v, c)
		                                                  for k, v in (get_annotations(c) | {k: get_annotations(v)['return']
		                                                                                     for k, v in vars(c).items()
		                                                                                     if isinstance(v, (property, functools.cached_property))
		                                                                                        and 'return' in get_annotations(v)
		                                                                                    }).items()
		                                                 } for c in S((*bases, *(j for i in bases for j in i.mro()))).uniquize()[::-1]), {}).items():
			if (v is not ... and (a := annotations.get(k)) and a != v and not ((isinstance(a, type) or typing_inspect.is_generic_type(a)) and typecheck(a, type[v]))):
				raise TypeError(f"Specified slot type annotation in class {name!r} ({k}: {format_inspect_annotation(a)}) is not a subclass of its parent's annotation in class {c.__name__!r} ({k}: {format_inspect_annotation(v)})")

		return super().__new__(metacls, name, bases, classdict)
class SlotsTypecheck(metaclass=SlotsTypecheckMeta): pass
class ABCSlotsTypecheckMeta(SlotsTypecheckMeta, abc.ABCMeta): pass
class ABCSlotsTypecheck(metaclass=ABCSlotsTypecheckMeta): pass

class TypeInitMeta(SlotsTypecheckMeta):
	def __new__(metacls, name, bases, classdict):
		cls = super().__new__(metacls, name, bases, classdict)
		if (not get_annotations(cls)): return cls

		__init_o__ = cls.__init__

		@suppress_tb
		def __init__(self, *args, **kwargs):
			for k, v in get_annotations(cls).items():
				if (v is ... or isinstance(v, str)): continue
				if (isinstance(getattr(cls, k, None), (property, functools.cached_property))): continue
				try: object.__getattribute__(self, k)
				except AttributeError: object.__setattr__(self, k, (v() if (isinstance(v, (type, function, method, builtin_function_or_method, types.GenericAlias))) else v))
			__init_o__(self, *args, **kwargs)

		cls.__init__ = __init__
		#cls.__metaclass__ = metacls  # inheritance?

		return cls
class TypeInit(metaclass=TypeInitMeta): pass
class ABCTypeInitMeta(TypeInitMeta, abc.ABCMeta): pass
class ABCTypeInit(metaclass=ABCTypeInitMeta): pass

class SlotsMeta(SlotsOnlyMeta, TypeInitMeta): pass
class Slots(metaclass=SlotsMeta): pass
class ABCSlotsMeta(SlotsMeta, abc.ABCMeta): pass
class ABCSlots(metaclass=ABCSlotsMeta): pass

class SlotsInitMeta(SlotsOnlyMeta):
	def __new__(metacls, name, bases, classdict):
		cls = super().__new__(metacls, name, bases, classdict)
		#if (not get_annotations(cls)): return cls

		@suppress_tb
		def __init__(self, *args, **kwargs):
			for k, v in allannotations(cls).items():
				try: a = kwargs.pop(k)
				except KeyError:
					if (typing_inspect.is_optional_type(v) or (hasattr(types, 'UnionType') and isinstance(v, types.UnionType) and NoneType in typing.get_args(v))): a = (typing.get_origin(v := typing.get_args(v)[-1]) or v)()
					else: raise TypeError(f"{try_repr(self)} missing a required keyword-only argument: {k}")
				else:
					if (isinstance(v, type) and not isinstance(a, v)): raise TypeError(f"{try_repr(a)} is not of type {try_repr(v)}")
				object.__setattr__(self, k, a)

		if ('__init__' not in classdict): cls.__init__ = __init__
		#cls.__metaclass__ = metacls  # inheritance?

		return cls
class SlotsInit(metaclass=SlotsInitMeta): pass
class ABCSlotsInitMeta(SlotsInitMeta, abc.ABCMeta): pass
class ABCSlotsInit(metaclass=ABCSlotsInitMeta): pass

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

class StoppableThread(threading.Thread):
	def __init__(self,  *args, kwargs=None, **kwargs_):
		_stop_event = self._stop_event = threading.Event()
		if (kwargs is None): kwargs = dict()
		kwargs['_stop_event'] = _stop_event
		super().__init__(*args, kwargs=kwargs, **kwargs_)

	def stop(self):
		self._stop_event.set()

	def is_stopped(self):
		return self._stop_event.is_set()

@funcdecorator
def instantiate(f):
	""" Decorator that replaces type returned by decorated function with instance of that type and leaves it as is if it is not a type. """
	def decorated(*args, **kwargs):
		r = f(*args, **kwargs)
		return r() if (isinstance(r, type)) else r
	return decorated

class aiobject:
	async def __new__(cls, *args, **kwargs):
		instance = super().__new__(cls)
		await instance.__init__(*args, **kwargs)
		return instance

	async def __init__(self):
		pass

def noop(*_, **__): pass
@singleton
class noopf:
	__call__ = noop

	def __repr__(self):
		return 'noopf'

	def __bool__(self):
		return False
@singleton
class noopcm:
	def __repr__(self):
		return 'noopcm'

	def __bool__(self):
		return False

	def __enter__(*_):
		pass

	def __exit__(*_):
		pass

@staticitemget
def each(*x): pass

@funcdecorator
def apmain(f):
	def decorated(*, nolog=None):
		if (nolog is None): nolog = not any(log._logged_start)
		if (hasattr(f, '_argdefs')): # TODO: move out of `decorated()`?
			while (f._argdefs):
				args, kwargs = f._argdefs.popleft()
				argparser.add_argument(*args, **kwargs)
		argparser.set_defaults(func=lambda *_: sys.exit(argparser.print_help()))
		cargs = argparser.parse_args()
		if (not nolog): logstarted()
		return f(cargs)
	return decorated

@dispatch
def apcmd(f: callable): return apcmd()(f)
@dispatch
def apcmd(*args, **kwargs):
	@funcdecorator(suppresstb=False)
	def decorator(f):
		nonlocal args, kwargs
		subparser = _getsubparser(*args, **kwargs).add_parser(f.__name__.removesuffix('_'), help=f.__doc__)
		if (hasattr(f, '_argdefs')):
			for args, kwargs in f._argdefs:
				subparser.add_argument(*args, **kwargs)
		subparser.set_defaults(func=f)
		return f
	return decorator
@cachedfunction
def _getsubparser(*args, **kwargs): return argparser.add_subparsers(*args, **kwargs)

def aparg(*args, **kwargs):
	@funcdecorator(suppresstb=False)
	def decorator(f):
		try: argdefs = f._argdefs
		except AttributeError: argdefs = f._argdefs = collections.deque()
		argdefs.appendleft((args, kwargs))
		return f
	return decorator

@funcdecorator
def asyncrun(f):
	def decorated(*args, **kwargs):
		return asyncio.run(f(*args, **kwargs))
	return decorated

class VarInt(Singleton):
	@staticmethod
	def read(s):
		r = int()
		i = int()
		while (True):
			b = s.read(1)[0]
			r |= ((b & 0x7f) << (7*i))
			if (not b & 0x80): break
			i += 1
		if (r & 1): r = -r
		return (r >> 1)

	@staticmethod
	def pack(v):
		r = bytearray()
		c, v = (v < 0), (abs(v) << 1)
		if (c): v -= 1
		while (True):
			c |= (v & 0x7f)
			v >>= 7
			if (v): c |= 0x80
			r.append(c)
			if (not v): break
			c = 0
		return bytes(r)

class hashabledict(dict):
	__setitem__ = \
	__delitem__ = \
	clear = \
	pop = \
	popitem = \
	setdefault = \
	update = classmethod(suppress_tb(lambda cls, *args, **kwargs: raise_(TypeError(f"'{cls.__name__}' object is not modifiable"))))

	def __hash__(self):
		return hash(tuple(self.items()))

class paramset(set):
	def __init__(self, x=()):
		super().__init__(map(str, x))

	def __hash__(self):
		return hash(tuple(sorted(self)))

	def __contains__(self, x):
		return super().__contains__(str(x))

	def __getattr__(self, x):
		return str(x) in self
	__getitem__ = __getattr__

	def __setattr__(self, x, y):
		if (y): self.add(x)
		else: self.discard(x)
	__setitem__ = __setattr__

	def __delattr__(self, x):
		self.discard(x)
	__delitem__ = __delattr__

	def add(self, x):
		return super().add(str(x))

	def discard(self, x):
		return super().discard(str(x))

	def update(self, x):
		return super().update(map(str, x))

class listmap:
	def __init__(self):
		self._keys = collections.deque()
		self._values = collections.deque()

	def __repr__(self):
		return ', '.join(f"{k}: {v}" for k, v in zip(self._keys, self._values)).join('[]')

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
	class _empty: __slots__ = ()

	def __init__(self, list_=None):
		self._list = list_ or []

	def __repr__(self):
		return f"indexset({try_repr(dict(enumerate(self._list)))})"

	def __getitem__(self, x):
		try: return self._list.index(x)
		except ValueError:
			self._list.append(x)
			return len(self._list)-1

	def __delitem__(self, x):
		self._list[x] = self._empty

	def __iter__(self):
		return iter(self._list)

	def copy(self):
		return indexset(self._list.copy())

	@itemget
	def values(self, x):
		try:
			if (self._list[x] is self._empty): raise KeyError(x)
			return self._list[x]
		except IndexError: pass
		raise KeyError(x)

class hashset(Slots):
	hashes: dict

	class _Getter(Slots):
		hash: ...

		def __init__(self, hash):
			self.hash = hash

		def __eq__(self, x):
			return hash(x) == self.hash

	def __getattr__(self, x):
		return self._Getter(self.hashes.get(x))

	def __setattr__(self, x, v):
		self.hashes[x] = hash(v)

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

### XXX?
#def lstripcount(s, chars=string.whitespace):
#	for ii, i in enumerate(s):
#		if (i not in chars): break
#	else: ii = 0
#	return (ii, s[ii:])
###

def lstripcount(s, chars=None):
	ns = s.lstrip(chars)
	return (len(s)-len(ns), ns)

def Sexcepthook(exctype=None, exc=None, tb=None, *, file=None, linesep='\n'): # TODO: bpo-43914
	if (exctype is None): exctype, exc, tb = sys.exc_info()
	else:
		if (exc is None): exc, exctype = exctype, type(exctype)
		if (tb is None): tb = exc.__traceback__
	if (file is None): file = sys.stderr
	_linesep = linesep

	srcs = Sdict(linecache.getlines)
	res = list()

	if (__repl__):
		if (exctype is SyntaxError and exc.text and exc.text.strip()[-1:] == '?'):
			topic = exc.text.strip()[:-1]
			try: help(eval(topic))
			except Exception:
				try: help(topic)
				except Exception: pass
				else: return
			else: return

	if (exc is not None and exc.__context__ is not None and not exc.__suppress_context__):
		Sexcepthook(exc.__context__, file=file, linesep=_linesep)
		print(" \033[0;1m> During handling of the above exception, another exception occurred:\033[m\n", file=file)

	for frame, lineno in traceback.walk_tb(tb):
		code = frame.f_code
		if (os.path.basename(code.co_filename) == 'runpy.py' and os.path.dirname(code.co_filename) in sys.path): continue

		filename = code.co_filename
		name = code.co_name
		src = srcs[code.co_filename]
		lines = set()
		codepos = (first(itertools.islice(code.co_positions(), frame.f_lasti//2, None), default=None) if (sys.version_info >= (3, 11)) else None)

		if (src and lineno is not None and lineno > 0):
			loff = float('inf')
			found_name = bool()

			lastline = None
			for i in range(lineno-1, 0, -1): #frame.f_lineno
				try: line = src[i]
				except IndexError: continue
				if (line.isspace()): continue
				cloff = lstripcount(line)[0]
				if (cloff < loff or i+1 == lineno or re.match(r'^\s*(elif|else|except|finally)\b', lastline)):
					#if (cloff > loff): continue  # show only same-level constructs for continuator clauses
					loff = cloff
					try: hline = highlight(line)
					except Exception: hline = line
					if (not found_name and name.isidentifier() and re.fullmatch(fr"\s*(?:def|class)\s+{name}\b.*(:|\(|\\)\s*(?:#.*)?", line)):
						hline = re.sub(fr"((?:def|class)\s+)({name})\b", '\\1\033[0;93m\\2\033[0;2m', hline, 1)
						found_name = True
					lines.add((i+1, line, hline))
					lastline = line

			#for i in range(*sorted((frame.f_lineno-1, lineno))):
			#	lines.add((i+1, src[i]))

		res.append((filename, name, lineno, sorted(lines), frame, codepos))

	if (res):
		print("\033[0;91mTraceback\033[m \033[2m(most recent call last)\033[m:", file=file)
		maxlnowidth = max((max(len(str(ln)) for ln, line, hline in lines) for filename, name, lineno, lines, frame, codepos in res if lines), default=0)

	last = lines = lastlines = None
	repeated = int()
	for filename, name, lineno, lines, frame, codepos in res:
		if (os.path.commonpath((os.path.abspath(filename), os.getcwd())) != '/'): filename = os.path.relpath(filename)

		if ((filename, lineno) != last):
			if (repeated > 1): print(f" \033[2;3m(last frame repeated \033[1m{repeated}\033[0;2;3m more times)\033[m\n", file=file); repeated = int()

			filepath = (os.path.dirname(filename)+os.path.sep if (os.path.dirname(filename)) else '')
			link = (f"file://{socket.gethostname()}"+os.path.realpath(filename) if (os.path.exists(filename)) else filename)
			if (lines or (lineno is not None and lineno > 0)): print('  File '+terminal_link(link, f"\033[2;96m{filepath}\033[0;96m{os.path.basename(filename)}\033[m")+f", in \033[93m{terminal_link(repr(frame.f_globals[name]), name) if (name in frame.f_globals) else name}\033[m, line \033[94m{lineno}\033[m{':'*bool(lines)}", file=file)
			else: print('  \033[2mFile '+terminal_link(link, f"\033[36m{filepath}\033[96m{os.path.basename(filename)}\033[0;2m")+f", in \033[93m{name}\033[0;2m, line \033[94m{lineno}\033[m", file=file)

			mlw = int()
			for ii, (ln, line, hline) in enumerate(lines):
				mlw = max(mlw, len(line.expandtabs(4)))
				print(end=' '*(8+(maxlnowidth-len(str(ln)))), file=file)
				#if (ln == lineno): print(end='\033[1m', file=file)  # bold lineno
				if (ii != len(lines)-1): print(end='\033[2m', file=file)
				print(ln, end='\033[m ', file=file)
				print('\033[2m│', end='\033[m ', file=file)
				if (ln == lineno): hc = 1
				#elif (ii != len(lines)-1): hc = 2  # dark context lines
				else: hc = None
				if (hc is not None):
					print(end=f"\033[{hc}m", file=file)
					hline = hline.replace(r';00m', rf";00;{hc}m") # TODO FIXME \033
				print(hline.rstrip().expandtabs(4), end='\033[m\n', file=file) #'\033[m'+ # XXX?
				if (codepos is not None and codepos[0] == ln):
					nt, sl = S(line).lstripcount('\t')
					if (codepos[3] < len(sl)): print(' '*(8+maxlnowidth+3 - (codepos[2] <= 1)), ' '*(4*nt), ' '*(codepos[2] - nt - 1), *(('\033[2;95m', '╰', '\033[22m', '╌'*(codepos[3] - codepos[2] + (codepos[2] == 1)), '\033[2m', '╯') if (codepos[3] - codepos[2] > 1+2) else ' \033[2;95m^\033[m'), sep='', end='\033[m\n', file=file)

		else:
			#if (lines == lastlines): repeated += 1 # TODO FIXME XXX: scope
			if (lines != lastlines or repeated <= 1):
				if ((lines or (lineno is not None and lineno > 0)) and not lastlines and _linesep is not None): print(end=_linesep, file=file)
				print("    \033[2m..."+('\033[m'*bool(lines or (lineno is not None and lineno > 0)))+f"in \033[93m{name}\033[m{':'*bool(lines)}", file=file)
				if (repeated > 1): print(f" \033[2;3m(last frame repeated \033[1m{repeated}\033[0;2;3m more times)\033[m\n", file=file); repeated = int()
		last = (filename, lineno)

		if (lines and repeated < 2):
			#try: c = compile(lines[-1][1].strip(), '', 'exec')
			#except SyntaxError:
			c = frame.f_code

			words = list()
			words_line = set()
			for ii, l in enumerate(lines, 1):
				for w in regex.findall(r'(?<=^|[^.[])(\w+|\.\w+|\[[\w-]+\]?)', l[1]):
					if (words and (w.startswith('.') or w.startswith('['))): w = (words[-1] + w) #words[-1] += w
					words.append(w)
					if (ii == len(lines)): words_line.add(w)

			inaccessible = set()
			for w in S(words).uniquize():
				if (keyword.iskeyword(w)): continue

				v = None
				#if (any(map(w.startswith, inaccessible))): v = '\033[3;90m<inaccessible>\033[m'
				for color, ns in ((93, frame.f_locals), (92, frame.f_globals), (95, builtins.__dict__)):
					#try: r = S(repr(obj := operator.attrgetter(w)(S(ns)))).indent(first=False)
					try:
						try:
							obj = Sdict(ns)
							for i in w.split('.'):
								obj = eval('_.'+i, {'_': obj})
						except (SyntaxError, AttributeError): continue
						else: r = S(repr(obj)).noesc()
					except Exception as ex:
						if (any(map(w.startswith, inaccessible))): continue
						inaccessible.add(w)
						color, r = 91, S(f"<exception in {w}.__repr__(): {S(try_repr(ex)).noesc()}>; {S(try_repr(obj)).noesc()}")
					if (r == w): continue

					if ((rf := r.fit(mlw-len(w)-1)) != r): r = S(terminal_link(r, rf))
					v = f"\033[{color}m{r.indent(12, first=False)}\033[m"
					break
				else:
					if (w.replace('.', '').isidentifier() and w not in builtins.__dict__):
						v = '\033[90m<not found>\033[m'
						try: del obj
						except NameError: pass

				if (v is not None):
					try: name = terminal_link(obj.__name__ + format_inspect_signature(_inspect_signature(obj, eval_str=True)), w)
					except Exception:
						try: name = terminal_link(str(type(obj)), w)
						except Exception: name = w
					print(f"{' '*12}\033[{'2;'*(w not in words_line)}{'2;92' if (ns is frame.f_locals and w in frame.f_code.co_varnames[:frame.f_code.co_argcount]) else '94'}m{name}\033[0;2m:\033[m \033[2m{v}", file=file)

			if (_linesep is not None): print(end=_linesep, file=file)
		elif (not lastlines and _linesep is not None): print(end=_linesep, file=file)
		lastlines = lines

	if (repeated): print(f" \033[2;3m(last frame repeated \033[1m{repeated}\033[0;2;3m times)\033[m\n", file=file); repeated = int() # TODO FIXME needed?

	if (exctype is KeyboardInterrupt and not exc.args): print(f"\033[0;2m{exctype.__name__}\033[m", file=file)
	elif (exctype is SyntaxError and exc.args):
		try: line = highlight(exc.text)
		except Exception: line = exc.text
		print(f"\033[0;1;96m{exctype.__name__}\033[m: {exc}", file=file)
		if (line is not None): print(f"{line.rstrip().expandtabs(1)}\n{' '*(exc.offset-1)}\033[95m{'^'*max(1, exc.end_offset - exc.offset)}\033[m", file=file)
	elif (exc is not None):
		try: s = str(exc)
		except Exception: s = "\033[2;3m (\033[91m<exception str() failed>\033[39m)\033[m"
		else:
			if (s): s = f": {s}"
		print(f"\033[0;1;91m{exctype.__name__}\033[m{s}", file=file)

	for i in getattr(exc, '__notes__', ()):
		print(f"  \033[2m{i}\033[m", file=file)

	if (exc is not None and exc.__cause__ is not None):
		print("\n \033[0;1m> This exception was caused by:\033[m\n", file=file)
		Sexcepthook(exc.__cause__, file=file, linesep=_linesep)

	if (__repl__ and tb is not None and tb.tb_frame.f_code.co_filename == '<stdin>'):
		if (exctype is NameError and exc.args and
		    (m := re.fullmatch(r"name '(\w+)' is not defined", exc.args[0])) is not None and
		    (module := importlib.util.find_spec(m[1])) is not None):
			print(f"\n\033[0;96m>>> \033[2mimport \033[1m{module.name}\033[m", file=file)
			frame.f_globals[m[1]] = module.loader.load_module()
			#readline.insert_text(readline.get_history_item(readline.get_current_history_length())) # TODO
		elif (exctype is AttributeError and exc.args and
		      (m := re.fullmatch(r"module '(\w+)(.+)?' has no attribute '(\w+)'", exc.args[0])) is not None and
		      (module := importlib.util.find_spec(m[1]+(m[2] or '')+'.'+m[3])) is not None):
			print(f"\n\033[0;96m>>> \033[2mimport \033[1m{module.name}\033[m", file=file)
			setattr((operator.attrgetter(m[2].removeprefix('.'))(frame.f_globals[m[1]]) if (m[2]) else frame.f_globals[m[1]]), m[3], module.loader.load_module())
			#readline.insert_text(readline.get_history_item(readline.get_current_history_length())) # TODO

	print(end='\033[m', file=file, flush=True)

def _Sexcepthook_install():
	if (sys.excepthook is not Sexcepthook):
		if (not hasattr(sys, '_S_oldexcepthook')): sys._S_oldexcepthook = sys.excepthook
		sys.excepthook = Sexcepthook
Sexcepthook.install = _Sexcepthook_install
if (sys.stderr.isatty()): Sexcepthook.install()

def getsrc(x, *, color=None, clear_term=True, ret=False):
	if (color is None): color = (not ret and sys.stdout.isatty())
	if (clear_term): clear()
	r = inspect.getsource(x)
	if (color): r = highlight(r)
	if (ret): return r
	else: print(r)

def preeval(f):
        r = f()
        return lambda: r

class grep(Slots):
	expr: ...
	flags: ...
	sep: ...

	@init(sep='\n')
	def __init__(self, expr, flags=0):
		self.expr, self.flags = repr(expr), flags

	def __ror__(self, x):
		for l in Sstr(x).noesc().split(self.sep):
			m = re.search(self.expr, l, self.flags)
			if (m is None): continue
			print(re.sub(self.expr.join('()'), '\033[1;91m\\1\033[m', l))

#def printf(format, *args, file=sys.stdout, flush=False): print(format % args, end='', file=file, flush=flush)  # breaks convenience of 'pri-' tab-completion.
class _CStream: # because I can.
	@dispatch
	def __init__(self, fd):
		self.ifd = self.ofd = fd

	@dispatch
	def __init__(self, ifd, ofd):
		self.ifd = ifd
		self.ofd = ofd

	def __repr__(self):
		return ('\033[F' if (sys.flags.interactive) else super().__repr__())
class IStream(_CStream):
	def __rshift__(self, x):
		globals = inspect.stack(0)[1].frame.f_globals
		globals[x] = type(globals.get(x, ''))(self.ifd.readline().rstrip('\n')) # obviousity? naaooo
		return self
class OStream(_CStream):
	def __lshift__(self, x):
		print(x, end='', file=self.ofd, flush=(x is endl))
		return self
class IOStream(IStream, OStream): pass
cin = IStream(sys.stdin)
cout = OStream(sys.stdout)
cerr = OStream(sys.stderr)
cio = IOStream(sys.stdin, sys.stdout)

class UserError(Exception): pass
class WTFException(AssertionError, NotImplementedError): pass
class TODO(NotImplementedError): pass
class TEST(BaseException): pass
class NonLoggingException(Exception): pass

def exit(c=None, code=None, raw=False, nolog=None):
	if (nolog is None): nolog = not any(log._logged_start)
		#name = inspect.stack(0)[1].frame.f_globals.get('__name__')
		#nolog = not (name is not None and log._logged_start.get(name))
	if (not nolog and sys.stderr.isatty()): print('\r\033[K', file=sys.stderr, flush=True)
	unlocklog()
	db.save(nolog=True)
	if (not nolog): log(f"{c}" if (raw) else f"Exit: {c}" if (c and type(c) == str or hasattr(c, 'args') and c.args) else "Exit.")
	if (logfile): logfile.flush()
	try: sys.exit(int(bool(c)) if (code is not None) else code)
	finally:
		if (not nolog): log(raw=True)

def setonsignals(f=exit):
	#signal.signal(signal.SIGINT, f)
	signal.signal(signal.SIGTERM, f)

logstart('Utils')
if (__name__ == '__main__'):
	testprogress()
	log("\033[mWhy\033[0;2m are u trying to run me?! It \033[0;93mtickles\033[0;2m!..\033[m", raw=True)
else: logimported()

# by Sdore, 2021-24
#   www.sdore.me
