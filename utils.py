#!/usr/bin/python3
# Utils lib
#type: ignore

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
		try: inspect.stack()[1][0].f_globals[object.__getattribute__(self, '_as')] = module
		except (TypeError, IndexError): pass
		return getattr(module, x)

def Simport(x, as_=None):
	inspect.stack()[1][0].f_globals
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
	'pdb',
	'pip',
	'pty',
	'sys',
	'tty',
	'bson',
	'code',
	'copy',
	'dill',
	'glob',
	'gzip',
	'html',
	'http',
	'json',
	'math',
	'stat',
	'time',
	'toml',
	'uuid',
	'yaml',
	'zlib',
	'errno',
	'queue',
	'regex',
	'shlex',
	'types',
	'atexit',
	'base64',
	'bidict',
	'bisect',
	'codeop',
	'ctypes',
	'getkey',
	'locale',
	#'parser',
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
	'urllib',
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
	'urllib3',
	'zipfile',
	'aiocache',
	'aiofiles',
	'argparse',
	#'attrdict',
	'builtins',
	'datetime',
	'operator',
	'platform',
	'pygments',
	'readline',
	'requests',
	'tempfile',
	'warnings',
	'fractions',
	'functools',
	'importlib',
	'ipaddress',
	'itertools',
	'linecache',
	'mimetypes',
	'netifaces',
	'pyparsing',
	'threading',
	'traceback',
	'wakeonlan',
	'contextlib',
	'subprocess',
	'collections',
	'rlcompleter',
	'unicodedata',
	'configparser',
	'aioprocessing',
	#'better_exchook',
	'typing_inspect',
	'multiprocessing_on_dill as multiprocessing',
	#'nonexistenttest'
)
for i in _imports: Simport(*i.split()[::2])
del i, Simport # TODO FIXME? (inspect.stack() is too slow)
globals().update(sys.modules)  # TODO: remove from list above?

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
__repl__ = bool(getattr(sys, 'ps1', sys.flags.interactive))

### [replaced by Sexcepthook]
#if (sys.stderr.isatty()):
#	try: better_exchook
#	except ModuleNotFoundError: pass
#	else:
#		sys._oldexcepthook = sys.excepthook
#		better_exchook.install()
###

argparser = argparse.ArgumentParser(conflict_handler='resolve', add_help=False)
argparser.add_argument('-v', action='count', help=argparse.SUPPRESS)
argparser.add_argument('-q', action='count', help=argparse.SUPPRESS)
cargs = argparser.parse_known_args()[0]
argparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
loglevel = (cargs.v or 0)-(cargs.q or 0)

# TODO: types.*
function = types.FunctionType
builtin_function_or_method = types.BuiltinFunctionType
method = types.MethodType
generator = types.GeneratorType
CodeType = types.CodeType
TracebackType = types.TracebackType
NoneType = type(None)
inf = float('inf')
nan = float('nan')
nl = endl = NL = ENDL = '\n'

def isiterable(x): return isinstance(x, typing.Iterable)
def isiterablenostr(x): return isiterable(x) and not isinstance(x, str)
def isnumber(x): return isinstance(x, numbers.Number)
def parseargs(kwargs, **args): args.update(kwargs); kwargs.update(args); return kwargs
def hex(x, l=2): return '0x%%0%dX' % l % x
def md5(x, n=1): x = x if (isinstance(x, bytes)) else str(x).encode(); return hashlib.md5(md5(x, n-1).encode() if (n > 1) else x).hexdigest()
def b64(x): return base64.b64encode(x if (isinstance(x, bytes)) else str(x).encode()).decode()
def ub64(x): return base64.b64decode(x).decode()
def randstr(n=16, *, caseless=False, seed=None): return str().join((random.Random(seed) if (seed is not None) else random).choices(string.ascii_lowercase if (caseless) else string.ascii_letters, k=n))
def try_repr(x):
	try: return repr(x)
	except Exception: return object.__repr__(x)
def safeexec():
	try:
		sys.stderr.write('\033[2m>>> ')
		sys.stderr.flush()
		return exec(sys.stdin.readline())
	finally: sys.stderr.write('\033[0m')
def export(x):
	globals = inspect.stack()[1][0].f_globals
	if ('__all__' not in globals): all = globals['__all__'] = list()
	elif (not isinstance(globals['__all__'], list)): all = globals['__all__'] = list(globals['__all__'])
	else: all = globals['__all__']
	all.append((x.__qualname__ if (hasattr(x, '__qualname__')) else x.__name__ if (hasattr(x, '__name__')) else x.__class__.__name__).rpartition('.')[-1])
	return x
def suppress_tb(f): f.__code__ = code_with(f.__code__, firstlineno=0, **{'linetable' if (sys.version_info >= (3, 10)) else 'lnotab': b''}); return f

def terminal_link(link, text=None): return f"\033]8;;{noesc.sub('', link)}\033\\{text if (text is not None) else link}\033]8;;\033\\"

try: pygments
except ModuleNotFoundError: pass
else: import pygments.lexers, pygments.formatters

#@dispatch
def highlight(s: str, *, lexer=None, formatter=None):
	if (lexer is None): lexer = pygments.lexers.PythonLexer()
	if (formatter is None): formatter = pygments.formatters.TerminalFormatter(bg='dark')
	return pygments.highlight(s, lexer, formatter)

def S(x=None, *, ni_ok=False):
	""" Convert `x' to an instance of corresponding S-type. """
	ex = None
	try: return Stuple(x) if (isinstance(x, generator)) else eval('S'+type(x).__name__)(x) if (not isinstance(x, S_type)) else x
	except NameError: ex = True
	if (ni_ok): return x
	if (ex): raise NotImplementedError("S%s" % type(x).__name__)

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

	def group(self, n): # TODO FIXME: (*(...),) -> tuple() everywhere
		return Slist((*(j for j in i if (j is not None)),) for i in itertools.zip_longest(*[iter(self)]*n))

	def flatten(self):
		return Slist(j for i in self if i for j in (i if (i and isiterable(i) and not isinstance(i, str)) else (i,)))

	def strip(self, s=None):
		l = self.copy()
		if (not isiterable(s) or isinstance(s, str)): s = (s,)
		return Slist(i for i in l if (i if (s is None) else i not in s))

	def filter(self, t):
		return Slist(i for i in self if type(i) == t)

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

	def fit(self, l, *, end='…'):
		return Sstr(self if (len(self) <= l) else self[:l-len(end)]+end if (l >= len(end)) else '')

	def cyclefit(self, l, n, *, sep=' '*8, start_delay=0, **kwargs):
		if (len(self) <= l): return self
		n = max(0, (n % (len(self)+len(sep)+start_delay))-start_delay)
		return Sstr((self+sep)[n:]+self[:n]).fit(l)

	def join(self, l, *, first='', last=None):
		l = tuple(map(str, l))
		r = (str.join(self, l[:-1])+((last if (isinstance(last, str)) else last[len(l) > 2]) if (last is not None) else self)+l[-1]) if (len(l) > 1) else l[0] if (l) else ''
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

def code_with(code, **kwargs):
	if (sys.version_info >= (3, 8)): return code.replace(**{'co_'*(not k.startswith('co_'))+k: v for k, v in kwargs.items()})
	return CodeType(*(kwargs.get(i, kwargs.get('co_'+i, getattr(code, 'co_'+i))) for i in ('argcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', 'lnotab', 'freevars', 'cellvars')))
# TODO: func_with()

def funcdecorator(df=None, /, *, signature=None, suppresstb=True): # TODO: __dict__?
	if (df is None): return lambda df: funcdecorator(df, signature=signature, suppresstb=suppresstb)

	@suppress_tb
	def ndf(f, *args, **kwargs):
		if (not isinstance(f, function)): return lambda nf: ndf(nf, f, *args, **kwargs)

		nf = df(f)
		if (nf is f): return f

		if (suppresstb): nf = suppress_tb(nf) # TODO: option to disable

		nf.__name__, nf.__qualname__, nf.__module__, nf.__doc__, nf.__annotations__ = \
		 f.__name__,  f.__qualname__,  f.__module__,  f.__doc__,  f.__annotations__

		if (signature is not None):
			if (isinstance(signature, str)): nf.__text_signature__ = signature
			else: nf.__signature__ = signature
			inspect.signature(nf)  # validate
		else: nf.__signature__ = inspect.signature(f)

		nf.__code__ = code_with(nf.__code__, name=f"<decorated '{f.__code__.co_name}'>" if (not f.__code__.co_name.startswith('<decorated ')) else f.__code__.co_name)

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

class dispatch:  # TODO FIXME: f(*, x=3)
	""" Decorator which implements function overloading (dispatching) by argument and return type decorations. """

	class __FillValue: __slots__ = ()
	__FillValue = __FillValue()

	class __Param:
		__slots__ = ('name', 'type', 'kind', 'required')

		def __init__(self, name, type, kind, required):
			self.name, self.type, self.kind, self.required = name, type, kind, required

		def __repr__(self):
			return f"<Parameter '{self.name}: {format_inspect_annotation(self.type)}' ({self.kind}, {'default' if (not self.required) else 'required'})>"

		@classmethod
		def from_inspect_parameter(cls, param):
			return cls(param.name, param.annotation, param.kind, (param.default is inspect._empty))

	__slots__ = ('__qualname__', '__origname__')
	_overloaded_functions = Sdict(dict)
	_overloaded_functions_retval = Sdict(dict)
	_overloaded_functions_docstings = Sdict(dict)

	def __init__(self, f):
		self.__origname__ = f.__name__
		#self.__module__ = f.__module__
		self.__qualname__ = f.__qualname__

		if (getattr(f, '__signature__', None) is ...): del f.__signature__ # TODO FIXME ???
		fsig = inspect.signature(f)
		params_annotation = tuple(map(self.__Param.from_inspect_parameter, fsig.parameters.values()))

		self._overloaded_functions[self.__qualname__][params_annotation] = f
		self._overloaded_functions_retval[self.__qualname__][params_annotation] = fsig.return_annotation
		self._overloaded_functions_docstings[self.__qualname__][fsig] = f.__doc__

		f.__origname__, f.__name__ = f.__name__, f"Overloaded {f.__name__}"
		f.__code__ = code_with(f.__code__, name=f"<overloaded '{f.__qualname__}' for {f.__origname__}{format_inspect_signature(fsig)}>")

		#print(f"\n\033[1;92m{self.__qualname__}()\033[0;1m:\033[0m")
		#for params in self._overloaded_functions[self.__qualname__]:
		#	print("     \N{bullet}", end='')
		#	for param in params:
		#		print(f"\t\033[1;93m{param.name}\033[0m: \033[1;94m{format_inspect_annotation(param.type)}\033[0m\n\t  \033[2m({param.kind.name}, {'default' if (not param.required) else 'required'})\033[0m\n")
		#	print()

	def __repr__(self):
		return f"<overloaded function '{self.__qualname__}'>"

	@suppress_tb
	def __call__(self, *args, **kwargs):
		f, params = self.__dispatch(*args, **kwargs)

		r = f(*args, **kwargs)

		if ((retval := self._overloaded_functions_retval[self.__qualname__][params]) is not inspect._empty):
			if (not self.__typecheck(r, retval)): raise DispatchError(f"Return value of type {type(r)} doesn't match the return annotation of the corresponding '{self.__qualname__}' signature: {self.__origname__}{_format_inspect_signature(params, retval)}")

		return r

	def __get__(self, obj, objcls):
		return lambda *args, **kwargs: self(obj, *args, **kwargs)

	def __getitem__(self, *t):
		return self._overloaded_functions[self.__qualname__][t]

	@suppress_tb
	def __dispatch(self, *args, **kwargs): # TODO FIXME: inspect.Signature.bind()?
		args = list(args)

		for params, func in self._overloaded_functions[self.__qualname__].items():
			i = int()

			no = True

			for a in args:
				if (i >= len(params)): break
				param = params[i]
				if (not self.__typecheck(a, param.type)): break
				if (param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)): i += 1
			else: no = False

			if (no): continue

			if (i < len(params) and params[i].kind == inspect.Parameter.VAR_POSITIONAL): i += 1

			no = True
			kw = {j.name: j for j in params[i:]}
			pkw = set()
			varkw = tuple(filter(lambda x: x.kind == inspect.Parameter.VAR_KEYWORD, kw.values()))

			for a, av in kwargs.items():
				try: ckw = kw[a]
				except KeyError:
					if (not varkw): break
					ckw = varkw[0]
				if (not self.__typecheck(av, ckw.type)): break
				pkw.add(a)
			else: no = False

			if (no): continue
			if (not varkw and ({k for k, v in kw.items() if v.required} - pkw)): continue

			return (func, params)

		try: return (self._overloaded_functions[self.__qualname__][()], ())
		except KeyError: pass

		raise DispatchError(f"Parameters {S(', ').join((*(format_inspect_annotation(type(i)) for i in args), *(f'{k}={format_inspect_annotation(type(v))}' for k, v in kwargs.items()))).join('()')} don't match any of '{self.__qualname__}' signatures:\n{S(overloaded_format_signatures(self.__qualname__, sep=endl)).indent(2, char=' ')}\n(called as: '{self.__origname__}({', '.join((*(S(try_repr(i)).fit(32) for i in args), *(S(f'{k}={try_repr(v)}').fit(32) for k, v in kwargs.items())))})')")

	@classmethod
	def __typecheck(cls, o, t):
		if (t is None or t is inspect._empty): return True
		if (o is cls.__FillValue or t is cls.__FillValue): return False

		if (isinstance(t, (function, builtin_function_or_method))): return bool(t(o))

		if (isinstance(t, typing.TypeVar)):
			if (not isinstance(o, typing_inspect.get_constraints(t))): return False
		else:
			if (not isinstance(o, typing_inspect.get_origin(t) or t)): return False

		if (args := typing_inspect.get_args(t)):
			if ((origin := typing_inspect.get_origin(t)) is not None and issubclass(origin, typing.Tuple) and isinstance(o, typing.Tuple)):
				if (not all(itertools.starmap(cls.__typecheck, itertools.zip_longest(o, args, fillvalue=cls.__FillValue)))): return False

			if (isinstance(o, typing.Iterable) and not isinstance(o, typing.Iterator)):
				if (not all(cls.__typecheck(i, args[0]) for i in o)): return False

		return True

	def format_signatures(self, sep='\n\n'):
		return sep.join(Sstr().join((self.__qualname__, format_inspect_signature(fsig), ':\n\b    '+doc if (doc) else '')) for fsig, doc in self._overloaded_functions_docstings[self.__qualname__].items())

	@property
	def __name__(self):
		return f"Overloaded {self.__origname__}"

	@property
	def __doc__(self):
		signatures = self.format_signatures()
		try: doc = self._overloaded_functions_docstings[self.__qualname__][()]
		except KeyError: return signatures
		else: return (doc + '\n\n' + signatures)

	@property
	def __signature__(self):
		return ...

	@property
	def __code__(self):
		return code_with(self.__call__.__code__, name=f"<overload handler of '{f.__qualname__}'>") # TODO move to __new__

@funcdecorator
def dispatch_meta(f):
	if (f.__doc__): dispatch._overloaded_functions_docstings[f.__qualname__][()] = f.__doc__
	return f

def format_inspect_signature(fsig): return _format_inspect_signature(fsig.parameters, fsig.return_annotation)
def _format_inspect_signature(parameters, return_annotation=inspect._empty):
	result = list()
	posonlysep = False
	kwonlysep = True

	for p in parameters.values():
		if (p.kind == inspect.Parameter.POSITIONAL_ONLY): posonlysep = True
		elif (posonlysep): result.append('/'); posonlysep = False
		if (p.kind == inspect.Parameter.VAR_POSITIONAL): kwonlysep = False
		elif (p.kind == inspect.Parameter.KEYWORD_ONLY and kwonlysep): result.append('*'); kwonlysep = False
		result.append(f"{'*' if (p.kind == inspect.Parameter.VAR_POSITIONAL) else '**' if (p.kind == inspect.Parameter.VAR_KEYWORD) else ''}{p.name}{f': {format_inspect_annotation(p.annotation)}' if (p.annotation is not inspect._empty) else ''}{f' = {repr(p.default)}' if (p.annotation is not inspect._empty and p.default is not inspect._empty) else f'={repr(p.default)}' if (p.default is not inspect._empty) else ''}")

	if (posonlysep): result.append('/')
	rendered = ', '.join(result).join('()')
	if (return_annotation is not inspect._empty): rendered += f" -> {format_inspect_annotation(return_annotation)}"
	return rendered

def format_inspect_annotation(annotation): return (annotation.__name__ if (isinstance(annotation, function)) else inspect.formatannotation(annotation))

def cast(*types): return lambda x: (t(i) if (not isinstance(i, t)) else i for t, i in zip(types, x))

@suppress_tb
def cast_call(f, *args, **kwargs):
	fsig = inspect.signature(f)
	try:
		args = [(v.annotation)(args[ii]) if (v.annotation is not inspect._empty and not isinstance(args[ii], v.annotation)) else args[ii] for ii, (k, v) in enumerate(fsig.parameters.items()) if ii < len(args)]
		kwargs = {k: (fsig.parameters[k].annotation)(v) if (k in fsig.parameters and fsig.parameters[k].annotation is not inspect._empty and not isinstance(v, fsig.parameters[k].annotation)) else v for k, v in kwargs.items()}
	except Exception as ex: raise CastError() from ex
	r = f(*args, **kwargs)
	return (fsig.return_annotation)(r) if (fsig.return_annotation is not inspect._empty) else r
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

	return lambda *args, **kwargs: f(*args, **S(kwargs) & {k: v() for k, v in f.__annotations__.items() if k not in kwargs and not isinstance(v, str)})

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
	class _noncached:
		__slots__ = ('value',)

		def __init__(self, value):
			self.value = value

	def __init__(self, f):
		self.f = f
		self._cached = dict()
		self._obj = None

	def __get__(self, obj, objcls):
		self._obj = obj
		return self

	def __call__(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		k = tohashable((args, kwargs))
		if (k not in self._cached):
			r = self.f(*args, **kwargs)
			if (not isinstance(r, self._noncached)): self._cached[k] = r
			else: return r.value
		return self._cached[k]

	def nocache(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		return self.f(*args, **kwargs)

	def is_cached(self, *args, **kwargs):
		if (self._obj is not None): args = (self._obj, *args)
		k = tohashable((args, kwargs))
		return (k in self._cached)

	def clear_cache(self):
		self._cached.clear()
class cachedclass(cachedfunction): pass

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

try: cachedproperty = functools.cached_property # TODO
except AttributeError:
	class cachedproperty:
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

@dispatch
def allsubclassdict(cls: type):
	""" Get name-class mapping for all subclasses of class `cls' (see `allsubclasses'). """
	return {i.__name__: i for i in allsubclasses(cls)}
@dispatch
def allsubclassdict(obj): return allsubclassdict(type(obj))

@dispatch
def allannotations(cls: type):
	""" Get annotations dict for all the MRO of class `cls' in right («super-to-sub») order. """
	return {k: v for i in cls.mro()[::-1] if hasattr(i, '__annotations__') for k, v in i.__annotations__.items()}
@dispatch
def allannotations(obj): return allannotations(type(obj))

@dispatch
def allslots(cls: type):
	""" Get slots tuple for all the MRO of class `cls' in right («super-to-sub») order. """
	return tuple(j for i in cls.mro()[::-1] if hasattr(i, '__slots__') for j in i.__slots__)
@dispatch
def allslots(obj): return allannotations(type(obj))

def spreadargs(f, okwargs, *args, **akwargs):
	fsig = inspect.signature(f)
	kwargs = S(okwargs) & akwargs
	try: kwnames = (*(i[0] for i in fsig.parameters.items() if (assert_(i[1].kind != inspect.Parameter.VAR_KEYWORD) and i[1].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))),)
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
			if (missing): raise TypeError(f"""{f.__name__}() missing {decline(len(missing), ('argument', 'arguments'), sep=' required keyword-only ')}: {S(', ').join((i.join("''") for i in missing), last=(' and ', ', and '))}""")
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
loglock = queue.LifoQueue()
_logged_start = dict()
_logged_utils_start = None
def log(l=None, *x, sep=' ', end='\n', ll=None, raw=False, tm=None, format=False, width=80, unlock=False, flush=True, nolog=False, nofile=False): # TODO: finally rewrite me as class pls
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
		flush: if true, flush logfile if written to it.
		nolog: if true, force suppress printing to stderr.
		nofile: if true, force suppress printing to logfile.
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
	if (not unlock and not loglock.empty()): loglock.put(((_l, *_x), dict(sep=sep, end=end, raw=raw, tm=tm, nolog=nolog, nofile=nofile))); return clearstr
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
	if (logfile and not nofile):
		logfile.write(logstr)
		if (flush): logfile.flush()
	if (not nolog and (l or 0) <= loglevel): logoutput.write(logstr); logoutput.flush()
	return clearstr
def plog(*args, **kwargs): parseargs(kwargs, format=True); return log(*args, **kwargs)
def _dlog(*args, **kwargs): parseargs(kwargs, ll='\033[95m[\033[1mDEBUG\033[0;95m]\033[0;96m', tm=''); return log(*args, **kwargs)
def dlog(*args, **kwargs): return _dlog(*map(str, args), **kwargs)
def dplog(*args, **kwargs): parseargs(kwargs, format=True, sep='\n'); return _dlog(*args, **kwargs)
def rlog(*args, **kwargs): parseargs(kwargs, raw=True); return log(*args, **kwargs)
def logdumb(**kwargs): return log(raw=True, end='', **kwargs)
def logstart(x):
	""" from utils[.nolog] import *; logstart(name) """
	global _logged_start, _logged_utils_start
	if (_logged_utils_start is None): _logged_utils_start = False; return
	#if ((name := inspect.stack()[1][0].f_globals.get('__name__')) is not None):
	if (_logged_start.get(x) is True): return
	_logged_start[x] = True
	log(x+'\033[0m...', end=' ') #, nolog=(x == 'Utils'))
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
def setlogfile(f): global logfile; logfile = open(f, 'a') if (isinstance(f, str)) else f
def setloglevel(l): global loglevel; loglevel = l
def locklog(): loglock.put(None)
def unlocklog(): logdumb(unlock=True)
def logflush(): logdumb(flush=True)
def setutilsnologimport(): global _logged_utils_start; _logged_utils_start = True

_exc_handlers = set()
def register_exc_handler(f): _exc_handlers.add(f)
_logged_exceptions = set()
@dispatch
def exception(ex: BaseException, extra=None, *, once=False, raw=False, nolog=False, _caught=True):
	""" Log an exception.
	Parameters:
		ex: exception to log.
		extra: additional info to log.
		nolog: no not call exc_handlers.
	"""
	ex = unraise(ex)
	if (isinstance(ex, NonLoggingException)): return
	if (once):
		if (repr(ex) in _logged_exceptions): return
		_logged_exceptions.add(repr(ex))
	e = log(('\033[91m'+'Caught '*_caught if (not isinstance(ex, Warning)) else '\033[93m' if ('warning' in ex.__class__.__name__.casefold()) else '\033[91m')+ex.__class__.__qualname__+(' on line '+' -> '.join(terminal_link(f'file://{socket.gethostname()}'+os.path.realpath(i[0].f_code.co_filename) if (os.path.exists(i[0].f_code.co_filename)) else i[0].f_code.co_filename, i[1]) for i in traceback.walk_tb(ex.__traceback__) if i[1])).rstrip(' on line')+'\033[0m'+(': '+str(ex))*bool(str(ex))+('\033[0;2m; ('+str(extra)+')\033[0m' if (extra is not None) else ''), raw=raw)
	if (nolog): return
	for i in _exc_handlers:
		try: i(e, ex)
		except Exception: pass
def logexception(*args, **kwargs): return exception(*args, **kwargs, _caught=False)

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

class _clear:
	""" Clear the terminal. """

	def __call__(self):
		print(end='\033c', flush=True)

	def __repr__(self):
		self(); return ''
clear = _clear()

class DB:
	""" All-in-one lightweight database class. """

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
		with self.lock:
			globals = inspect.stack()[1][0].f_globals
			for field in fields:
				self.fields[field] = (globals, globals['__annotations__'][field], globals.get(field)) if (field in globals.get('__annotations__', ())) else globals

	def load(self, nolog=None):
		with self.lock:
			nolog = (self.nolog if (nolog is None) else nolog)

			if (not self.file): return
			if (not nolog): logstart('Loading database')

			db = dict()

			try: db = self.serializer.load(self.file)
			except EOFError:
				if (not nolog): logwarn('database is empty')
			else:
				if (not nolog): logok()

			self.file.seek(0)

			for field, globals in self.fields.items():
				if (isinstance(globals, tuple)): globals, annotation, default = globals
				try: value = db[field]
				except KeyError:
					if (not nolog): log(1, f"Not in DB: {field}")
				else: globals[field] = value

		return db

	def save(self, db={}, backup=None, nolog=None):
		with self.lock:
			nolog = (self.nolog if (nolog is None) else nolog)
			backup = (self.backup if (backup is None) else backup)

			if (not self.file): return
			if (not nolog): logstart('Saving database')

			if (backup):
				try: os.mkdir('backup')
				except FileExistsError: pass
				try: shutil.copyfile(self.file.name, f"backup/{self.file.name if (hasattr(self.file, 'name')) else ''}_{int(time.time())}.db")
				except OSError: pass

			try: self.serializer.dump(db or {field: globals[0][field] if (isinstance(globals, tuple)) else globals[field] for field, globals in self.fields.items() if not isinstance(globals, tuple) and field in globals or isinstance(globals, tuple) and field in globals[0] and globals[2] is not None and globals[0][field] != globals[2]}, self.file)
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

	def __del__(self):
		try:
			if (self.printed): sys.stderr.write('\n')
		except AttributeError: pass

	def format(self, cv, width, *, add_base=None, add_speed_eta=None):
		if (add_base is None): add_base = self.add_base
		if (add_speed_eta is None): add_speed_eta = self.add_speed_eta
		if (self.started is None): self.started = time.time(); add_speed_eta = False

		l = self.l(add_base) if (self._pool is None) else max(i.l(add_base) for i in self._pool.p)
		fstr = self.prefix+('%s/%s (%d%%%s) ' if (not self.fixed) else f"%{l}s/%-{l}s (%-3d%%%s) ")
		r = fstr % (*((cv, self.mv) if (not add_base) else map(self.format_base, (cv, self.mv)) if (add_base is True) else (self.format_base(i, base=add_base) for i in (cv, self.mv))), cv*100//self.mv, ', '+self.format_speed_eta(cv, self.mv, time.time()-self.started, fixed=self.fixed) if (add_speed_eta) else '')
		return r+self.format_bar(cv, self.mv, width-len(r), chars=self.chars, border=self.border, fill=self.fill)

	@staticmethod
	def format_bar(cv, mv, width, *, chars=' ▏▎▍▌▋▊▉█', border='│', fill=' '):
		cv = max(0, min(mv, cv))
		d = 100/(width-2)
		fp, pp = divmod(cv*100/d, mv)
		pb = chars[-1]*int(fp) + chars[int(pp*len(chars)//mv)]*(cv != mv)
		return f"{border}{pb.ljust(width-2, fill)}{border}"

	@staticmethod
	def format_speed_eta(cv, mv, elapsed, *, fixed=False):
		speed = cv/elapsed
		eta = math.ceil(mv/speed)-elapsed if (speed) else 0
		if (fixed): return (first(f"%2d{'dhms'[ii]}" % i for ii, i in enumerate((eta//60//60//24, eta//60//60%24, eta//60%60, eta%60)) if i) if (eta > 0) else ' ? ')+' ETA'
		eta = ' '.join(f"{int(i)}{'dhms'[ii]}" for ii, i in enumerate((eta//60//60//24, eta//60//60%24, eta//60%60, eta%60)) if i) if (eta > 0) else '?'
		speed_u = 0
		for i in (60, 60, 24):
			if (speed < 1): speed *= i; speed_u += 1
		return '%d/%c, %s ETA' % (speed, 'smhd'[speed_u], eta)

	@staticmethod
	def format_base(cv, base=(1000, ' KMB'), *, _calc_len=False):
		""" base: `(step, names)' [ex: `(1024, ('b', 'kb', 'mb', 'gb', 'tb')']
		names should be non-decreasing in length for correct use in fixed-width mode.
		whitespace character will be treated as nothing.
		"""

		step, names = base
		l = len(names)
		if (_calc_len): sl, nl = len(S(step))-1, max(map(len, names))
		try: return first((f"{{: {sl+nl}}}".format(v) if (_calc_len) else str(v))+(b if (b != ' ') else '') for b, v in ((i, cv//(step**(l-ii))) for ii, i in enumerate(reversed(names), 1)) if v) # TODO: fractions; negative
		except StopIteration: return '0'

	def print(self, cv, *, out=sys.stderr, width=None, flush=True):
		if (width is None): width = os.get_terminal_size()[0]
		out.write('\033[K'+self.format(cv, width=width)+'\r')
		if (flush): out.flush()
		self.printed = (out == sys.stderr)

	def l(self, add_base):
		return len(S(self.mv)) if (not add_base) else len(self.format_base(self.mv, _calc_len=True) if (add_base is True) else self.format_base(self.mv, base=add_base, _calc_len=True))

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
		self.p, self.kwargs = [Progress(-1, **parseargs(kwargs, fixed=True), _pool=self) for _ in range(n)], kwargs
		self.ranges = list()

	def __del__(self):
		try:
			for i in self.p:
				i.printed = False
		except AttributeError: pass
		sys.stderr.write('\033[J')
		sys.stderr.flush()

	def print(self, *cvs, width=None):
		ii = None
		for ii, (p, cv) in enumerate(zip(self.p, cvs)):
			if (ii): sys.stderr.write('\n')
			p.print(cv, width=width)
		if (ii): sys.stderr.write(f"\033[{ii}A")
		sys.stderr.flush()

	def range(self, start, stop=None, step=1, **kwargs):
		if (stop is None): start, stop = 0, start
		n = len(self.ranges)
		self.ranges.append(int())
		if (n == len(self.p)): self.p.append(Progress(stop-start, **parseargs(kwargs, **self.kwargs), _pool=self))
		else: self.p[n].mv = stop-start
		for i in range(start, stop, step):
			self.ranges[n] = i-start
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
		return self

	def __exit__(self, type, value, tb):
		self.stop()

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
		if (n == len(self.p)):
			self.p.append(Progress(stop-start, **parseargs(kwargs, **self.kwargs), _pool=self))
			self.cvs.append(int())
		else: self.p[n].mv = stop-start
		for i in range(start, stop, step):
			self.cvs[n] = i-start
			yield i
		self.ranges -= 1
		self.p.pop()

	def stop(self):
		self.stopped = True
		self.join()
		sys.stderr.write('\r\033[K')
		sys.stderr.flush()

def progrange(start, stop=None, step=1, **kwargs):
	parseargs(kwargs, fixed=True)
	if (stop is None): start, stop = 0, start
	p = Progress(stop-start, **kwargs)
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

def testprogress(n=1000, sleep=0.002, **kwargs):
	parseargs(kwargs, fixed=True)
	p = Progress(n, **kwargs)
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
		out.write(self.format(**fmtkwargs)+'\n')

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
		__slots__ = ('title', 'state', 'subtasks')

		def __init__(self, title, subtasks=None):
			self.title, self.subtasks = str(title), subtasks if (subtasks is not None) else []
			self.state = None

	def __init__(self, x, **kwargs):
		self.tree = x
		self.l = len(tuple(self.format_node(self.tree, root=True, **kwargs)))

	def __del__(self):
		sys.stderr.write('\n'*self.l)
		sys.stderr.flush()

	def print(self, **fmtkwargs):
		sys.stderr.write('\033[J')
		sys.stderr.write('\n'.join(x+' '+self.format_task(y) for x, y in self.format_node(self.tree, root=True, **fmtkwargs)))
		sys.stderr.write(f"\r\033[{self.l-1}A")
		sys.stderr.flush()

	def format_node(self, node, indent=2, color=False, *, root=False):
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
			r = eval(e.format(t(l[i]))) if (type(e) == str) else e(t(l[i]))
			if (bool(r) == False): raise ValidationError(r)
		except Exception as ex:
			if (not nolog): log(2, "\033[91mValidation error:\033[0m %s" % ex)
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
	return f"{p}{str(S(n).format(' ' if (format is True) else format) if (format) else n)+sep if (n != 1 or show_one) else ''}{r}"
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

def first(l): return next(iter(l))
def last(l):
	l = iter(l)
	r = next(l)
	while (True):
		try: r = next(l)
		except StopIteration: return r

def pm(x): return 1 if (x) else -1
def constrain(x, lb, ub): return min(ub, max(lb, x))
def prod(x, initial=1): return functools.reduce(operator.mul, x, initial)
def average(x, default=None): return sum(x)/len(x) if (x) else default if (default is not None) else raise_(ValueError("average() arg is an empty sequence"))

global_lambdas = list() # dunno why
def global_lambda(l): global_lambdas.append(l); return global_lambdas[-1]

@dispatch
@suppress_tb
def singleton(C: type): C.__new__ = cachedfunction(C.__new__); return C()
@dispatch
@suppress_tb
def singleton(*args, **kwargs): return lambda C: C(*args, **kwargs)

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

@singleton
class timecounter:
	def __init__(self):
		self.started = None
		self.ended = None

	def __call__(self):
		return self

	def __enter__(self):
		if (self is timecounter): self = type(self)()
		self.started = time.time()
		return self

	def __exit__(self, type, value, tb):
		self.ended = time.time()

	def time(self):
		if (self.started is None): return 0
		if (self.ended is None): return time.time()-self.started
		return self.ended-self.started

class classonlymethod:
	__slots__ = ('__func__',)

	def __init__(self, f):
		self.__func__ = f

	@suppress_tb
	def __get__(self, obj, cls=None):
		if (obj is not None): raise AttributeError(f"'{cls.__name__}.{self.__func__.__name__}()' is a class-only method.")
		return functools.partial(self.__func__, cls)

class classproperty:
	__slots__ = ('__func__',)

	def __init__(self, f):
		self.__func__ = f

	def __get__(self, obj, cls):
		return self.__func__(cls)

class attrget:
	__slots__ = ('f',)

	class getter:
		__slots__ = ('obj', 'f')

		def __init__(self, obj, f):
			self.obj, self.f = obj, f

		def __getattr__(self, x):
			return self.f(self.obj, x)

	def __init__(self, f):
		self.f = f

	def __get__(self, obj, cls):
		return self.getter(obj, self.f)

class itemget:
	__slots__ = ('f', '__call__', '_bool')

	class getter:
		__slots__ = ('obj', 'f')

		def __init__(self, obj, f):
			self.obj, self.f = obj, f

		def __contains__(self, x):
			try: self[x]
			except KeyError: return False
			else: return True

		def __getitem__(self, x):
			return self.f(self.obj, *(x if (isinstance(x, tuple)) else (x,)))

	@dispatch
	def __init__(self, *, bool: function):
		self._bool = bool
		self.__call__ = lambda f: each[setattr(self, 'f', f), delattr(self, '__call__')]

	@dispatch
	def __init__(self, f):
		self.f = f

	def __bool__(self):
		try: f = self._bool
		except AttributeError: return super().__bool__()
		else: return f(self.obj)

	def __get__(self, obj, cls):
		return self.getter(obj, self.f)

class classitemget(itemget):
	__slots__ = ('f',)

	def __init__(self, f):
		self.f = f

	def __get__(self, obj, cls):
		return self.getter(cls, self.f)

class staticitemget:
	__slots__ = ('f', '_fkeys')

	def __init__(self, f):
		self.f = f
		self._fkeys = lambda self: ()

	def __iter__(self):
		raise ValueError("staticitemget is not iterable")

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

class staticattrget:
	__slots__ = ('f',)

	def __init__(self, f):
		self.f = f

	def __getattr__(self, x):
		return self.f(x)

	def __call__(self, *args, **kwargs):
		return self.f(*args, **kwargs)

class AttrView:
	def __init__(self, obj):
		self.obj = obj

	def __repr__(self):
		return f"<AttrView of {self.obj}>"

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
		return f"<AttrProxy of {S(', ').join(self.objects)}>"

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

class SlotsOnlyMeta(type):
	def __new__(metacls, name, bases, classdict):
		annotations = classdict.get('__annotations__', {})
		classdict['__slots__'] = tuple(annotations.keys())
		return super().__new__(metacls, name, bases, classdict)
class SlotsOnly(metaclass=SlotsOnlyMeta): pass
class ABCSlotsOnlyMeta(SlotsOnlyMeta, abc.ABCMeta): pass
class ABCSlotsOnly(metaclass=ABCSlotsOnlyMeta): pass

class SlotsMeta(type):
	def __new__(metacls, name, bases, classdict):
		annotations = classdict.get('__annotations__', {})
		classdict['__slots__'] = tuple(annotations.keys())
		cls = super().__new__(metacls, name, bases, classdict)
		if (not annotations): return cls

		__init_o__ = cls.__init__
		@suppress_tb
		def __init__(self, *args, **kwargs):
			for k, v in annotations.items():
				if (v is ...): continue
				try: object.__getattribute__(self, k)
				except AttributeError: object.__setattr__(self, k, v() if (isinstance(v, (type, function, method))) else v)
			__init_o__(self, *args, **kwargs)

		cls.__init__ = __init__
		#cls.__metaclass__ = metacls  # inheritance?

		return cls
class Slots(metaclass=SlotsMeta): pass
class ABCSlotsMeta(SlotsMeta, abc.ABCMeta): pass
class ABCSlots(metaclass=ABCSlotsMeta): pass

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

@cachedfunction
def getsubparser(ap, *args, **kwargs): return ap.add_subparsers(*args, **kwargs)

@staticitemget
def each(*x): pass

@funcdecorator
def apmain(f):
	def decorated(*, nolog=None):
		if (nolog is None): nolog = not any(_logged_start)
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
	@funcdecorator(suppresstb=False)
	def decorator(f):
		nonlocal args, kwargs
		subparser = getsubparser(argparser, *args, **kwargs).add_parser(f.__name__.rstrip('_'), help=f.__doc__)
		if (hasattr(f, '_argdefs')):
			for args, kwargs in f._argdefs:
				subparser.add_argument(*args, **kwargs)
		subparser.set_defaults(func=f)
		return f
	return decorator

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

class VarInt:
	@staticmethod
	def read(s):
		r = int()
		i = int()
		while (True):
			b = s.read(1)[0]
			r |= (b & 0x7f) << (7*i)
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
	update = classmethod(lambda cls, *args, **kwargs: raise_(TypeError(f"'{cls.__name__}' object is not modifiable")))

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
	class _empty: pass

	def __init__(self, list_=None):
		self._list = list_ or []

	def __repr__(self):
		return f"indexset({repr(dict(enumerate(self._list)))})"

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

class hashset(metaclass=SlotsMeta):
	hashes: dict

	class _Getter(metaclass=SlotsMeta):
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

def Sexcepthook(exctype, exc, tb, *, file=None, linesep='\n'): # TODO: bpo-43914
	if (file is not None): _file = file
	else: _file = sys.stderr
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

	for frame, lineno in traceback.walk_tb(tb):
		code = frame.f_code
		if (os.path.basename(code.co_filename) == 'runpy.py' and os.path.dirname(code.co_filename) in sys.path): continue

		file = code.co_filename
		name = code.co_name
		src = srcs[code.co_filename]
		lines = set()

		if (src and lineno > 0):
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
					if (not found_name and re.fullmatch(fr"\s*(?:def|class)\s+{name}\b.*(:|\(|\\)\s*(?:#.*)?", line)):
						hline = re.sub(fr"((?:def|class)\s+)({name})\b", '\\1\033[0;93m\\2\033[0;2m', hline, 1)
						found_name = True
					lines.add((i+1, line, hline))
					lastline = line

			#for i in range(*sorted((frame.f_lineno-1, lineno))):
			#	lines.add((i+1, src[i]))

		res.append((file, name, lineno, sorted(lines), frame))

	if (res):
		print("\033[0;91mTraceback\033[0m \033[2m(most recent call last)\033[0m:", file=_file)
		maxlnowidth = max((max(len(str(ln)) for ln, line, hline in lines) for file, name, lineno, lines, frame in res if lines), default=0)

	last = lines = lastlines = None
	repeated = int()
	for file, name, lineno, lines, frame in res:
		if (os.path.commonpath((os.path.abspath(file), os.getcwd())) != '/'): file = os.path.relpath(file)

		if ((file, lineno) != last):
			if (repeated > 1): print(f" \033[2;3m(last frame repeated \033[1m{repeated}\033[0;2;3m more times)\033[0m\n", file=_file); repeated = int()

			filepath = (os.path.dirname(file)+os.path.sep if (os.path.dirname(file)) else '')
			link = f'file://{socket.gethostname()}'+os.path.realpath(file) if (os.path.exists(file)) else file
			if (lines or lineno > 0): print('  File '+terminal_link(link, f"\033[2;96m{filepath}\033[0;96m{os.path.basename(file)}\033[0m")+f", in \033[93m{name}\033[0m, line \033[94m{lineno}\033[0m{':'*bool(lines)}", file=_file)
			else: print('  \033[2mFile '+terminal_link(link, f"\033[36m{filepath}\033[96m{os.path.basename(file)}\033[0;2m")+f", in \033[93m{name}\033[0;2m, line \033[94m{lineno}\033[0m", file=_file)

			mlw = int()
			for ii, (ln, line, hline) in enumerate(lines):
				mlw = max(mlw, len(line.expandtabs(4)))
				print(end=' '*(8+(maxlnowidth-len(str(ln)))), file=_file)
				#if (ln == lineno): print(end='\033[1m', file=_file)  # bold lineno
				if (ii != len(lines)-1): print(end='\033[2m', file=_file)
				print(ln, end='\033[0m ', file=_file)
				print('\033[2m│', end='\033[0m ', file=_file)
				if (ln == lineno): hc = 1
				#elif (ii != len(lines)-1): hc = 2  # dark context lines
				else: hc = None
				if (hc is not None):
					print(end=f"\033[{hc}m", file=_file)
					hline = hline.replace(r';00m', rf";00;{hc}m") # TODO FIXME \033
				print(hline.rstrip().expandtabs(4), end='\033[0m\n', file=_file) #'\033[0m'+ # XXX?
		else:
			if (lines == lastlines): repeated += 1
			if (lines != lastlines or repeated <= 1):
				if ((lines or lineno > 0) and not lastlines and _linesep is not None): print(end=_linesep, file=_file)
				print("    \033[2m..."+('\033[0m'*bool(lines or lineno > 0))+f"in \033[93m{name}\033[0m{':'*bool(lines)}", file=_file)
				if (repeated > 1): print(f" \033[2;3m(last frame repeated \033[1m{repeated}\033[0;2;3m more times)\033[0m\n", file=_file); repeated = int()
		last = (file, lineno)


		if (lines and repeated < 2):
			#try: c = compile(lines[-1][1].strip(), '', 'exec')
			#except SyntaxError:
			c = frame.f_code

			words = list()
			for i in regex.findall(r'[^.]\b(\w+|\.\w+)\b', lines[-1][1], overlapped=True):
				if (words and i.startswith('.')): words.append(words[-1] + i) #words[-1] += i
				else: words.append(i)

			for i in S(words).uniquize():
				v = None
				for color, ns in ((93, frame.f_locals), (92, frame.f_globals), (95, builtins.__dict__)):
					try: r = S(repr(obj := operator.attrgetter(i)(S(ns)))).indent(first=False)
					except (KeyError, AttributeError): continue
					except Exception as ex: color, r = 91, S(f"<exception in {i}.__repr__(): {repr(ex)}>")
					else:
						if (r == i): continue
					rf = r.fit(mlw-len(i)-1)
					if (rf != r): r = terminal_link(r, rf)
					v = f"\033[{color}m{r}\033[0m"
					break
				else:
					if (i.replace('.', '').isidentifier() and not keyword.iskeyword(i) and i not in builtins.__dict__): v = '\033[90m<not found>\033[0m'

				if (v is not None):
					try: i = terminal_link(obj.__name__+format_inspect_signature(inspect.signature(obj)), i)
					except Exception:
						try: i = terminal_link(str(type(obj)), i)
						except Exception: pass
					print(f"{' '*12}\033[94m{i}\033[0;2m:\033[0m \033[2m{v}", file=_file)

			if (_linesep is not None): print(end=_linesep, file=_file)
		elif (not lastlines and _linesep is not None): print(end=_linesep, file=_file)
		lastlines = lines

	if (repeated): print(f" \033[2;3m(last frame repeated \033[1m{repeated}\033[0;2;3m times)\033[0m\n", file=_file); repeated = int() # TODO FIXME needed?

	if (exctype is KeyboardInterrupt and not exc.args): print(f"\033[0;2m{exctype.__name__}\033[0m", file=_file)
	elif (exctype is SyntaxError and exc.args):
		try: line = highlight(exc.text)
		except Exception: line = exc.text
		print(f"\033[0;1;96m{exctype.__name__}\033[0m: {exc}", file=_file)
		if (line is not None): print(f"{line.rstrip().expandtabs(1)}\n{' '*(exc.offset-1)}\033[95m^\033[0m", file=_file)
	else: print(f"\033[0;1;91m{exctype.__name__}\033[0m{': '*bool(str(exc))}{exc}", file=_file)

	if (exc.__cause__ is not None):
		print(" \033[0;1m> This exception was caused by:\033[0m\n", file=_file)
		Sexcepthook(type(exc.__cause__), exc.__cause__, exc.__cause__.__traceback__, file=_file, linesep=_linesep)

	if (exc.__context__ is not None and not exc.__suppress_context__):
		print(" \033[0;1m> During handling of the above exception, another exception occurred:\033[0m\n", file=_file)
		Sexcepthook(type(exc.__context__), exc.__context__, exc.__context__.__traceback__, file=_file, linesep=_linesep)

	if (__repl__ and tb is not None and tb.tb_frame.f_code.co_filename == '<stdin>'):
		if (exctype is NameError and exc.args and
		    (m := re.fullmatch(r"name '(.*)' is not defined", exc.args[0])) is not None and
		    (module := importlib.util.find_spec(m[1])) is not None):
			print(f"\n\033[0;96m>>> \033[2mimport \033[1m{module.name}\033[0m", file=_file)
			frame.f_globals[m[1]] = module.loader.load_module()
			#readline.insert_text(readline.get_history_item(readline.get_current_history_length())) # TODO
		elif (exctype is AttributeError and exc.args and
		      (m := re.fullmatch(r"module '(.*)' has no attribute '(.*)'", exc.args[0])) is not None and
		      (module := importlib.util.find_spec(m[1]+'.'+m[2])) is not None):
			print(f"\n\033[0;96m>>> \033[2mimport \033[1m{module.name}\033[0m", file=_file)
			setattr(frame.f_globals[m[1]], m[2], module.loader.load_module())
			#readline.insert_text(readline.get_history_item(readline.get_current_history_length())) # TODO
def _Sexcepthook_install():
	if (sys.excepthook is not Sexcepthook):
		if (not hasattr(sys, '_S_oldexcepthook')): sys._S_oldexcepthook = sys.excepthook
		sys.excepthook = Sexcepthook
Sexcepthook.install = _Sexcepthook_install
if (sys.stderr.isatty()): Sexcepthook.install()

def getsrc(x, *, color=True, clear_term=True, ret=False):
	if (clear_term): clear()
	r = inspect.getsource(x)
	if (color and sys.stderr.isatty()): r = highlight(r)
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
		self.expr, self.flags = expr, flags

	def __ror__(self, x):
		for l in noesc.sub('', str(x)).split(self.sep):
			m = re.search(self.expr, l, self.flags)
			if (m is None): continue
			print(re.sub(self.expr.join('()'), '\033[1;91m\\1\033[0m', l))

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
class WTFException(AssertionError, NotImplementedError): pass
class TODO(NotImplementedError): pass
class TEST(BaseException): pass
class NonLoggingException(Exception): pass

def exit(c=None, code=None, raw=False, nolog=None):
	global _logged_start
	if (nolog is None): nolog = not any(_logged_start)
		#name = inspect.stack()[1][0].f_globals.get('__name__')
		#nolog = not (name is not None and _logged_start.get(name))
	if (not nolog): sys.stderr.write('\r\033[K')
	unlocklog()
	db.save(nolog=True)
	if (not nolog): log(f'{c}' if (raw) else f'Exit: {c}' if (c and type(c) == str or hasattr(c, 'args') and c.args) else 'Exit.')
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
	log('\033[0mWhy\033[0;2m are u trying to run me?! It \033[0;93mtickles\033[0;2m!..\033[0m', raw=True)
else: logimported()

# by Sdore, 2021
