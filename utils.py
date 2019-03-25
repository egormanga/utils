#!/usr/bin/python3
# Utils lib

""" Sdore's Utils

Code example:

#!/usr/bin/python3
# <NAME>

from utils import *; logstart('<NAME>')

def main(): pass

if (__name__ == '__main__'): logstarted(); main()
else: logimported()

# by <AUTHOR>, <YEAR>
"""

import inspect
from pprint import pprint, pformat

module = type(inspect)

class ModuleProxy(module):
	def __init__(self, name, as_=None):
		self.__name__, self.as_ = name, as_ or name

	def __getattribute__(self, x):
		module = __import__(object.__getattribute__(self, '__name__'))
		inspect.stack()[1][0].f_globals[object.__getattribute__(self, 'as_')] = module
		return getattr(module, x)

def Simport(x, as_=None): inspect.stack()[1][0].f_globals[as_ or x] = ModuleProxy(x, as_)

for i in (
	'io',
	'os',
	're',
	'sys',
	'copy',
	'dill',
	'glob',
	'html',
	'json',
	'math',
	'queue',
	'regex',
	'time',
	'base64',
	'locale',
	'pickle',
	'random',
	'select',
	'shutil',
	'signal',
	'socket',
	'string',
	'typing',
	'getpass',
	'os.path',
	'argparse',
	'builtins',
	'datetime',
	'operator',
	'requests',
	'importlib',
	'itertools',
	'threading',
	'traceback',
	'contextlib',
	'subprocess',
	'collections',
	'typing_inspect',
	'multiprocessing_on_dill as multiprocessing',
	#'nonexistenttest',
): Simport(*i.split()[::2])

py_version = 'Python '+sys.version.split(maxsplit=1)[0]

argparser = argparse.ArgumentParser(conflict_handler='resolve', description='\033[3A', add_help=False)
argparser.add_argument('-v', action='count', help=argparse.SUPPRESS)
argparser.add_argument('-q', action='count', help=argparse.SUPPRESS)
cargs, _ = argparser.parse_known_args()
argparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
loglevel = (cargs.v or 0)-(cargs.q or 0)

endl = '\n'

def isiterable(x): return isinstance(x, typing.Iterable)
def isnumber(x): return isinstance(x, (int, float))
def parseargs(kwargs, **args): args.update(kwargs); kwargs.update(args)

def S(x=None):
		ex = None
		try: return eval('S'+type(x).__name__)(x) if (not isinstance(x, Stype)) else x
		except NameError: ex = True
		if (ex): raise \
			NotImplementedError("S%s" % type(x).__name__)

class Stype: pass

class Sdict(Stype, collections.defaultdict):
	def __init__(self, *args, **kwargs):
		args = list(args)
		if (args and isiterable(args[0])): args.insert(0, None)
		if (not args): args.insert(0, None)
		super().__init__(*args, **kwargs)

	def __and__(self, x):
		r = self.copy(); r.update(x); return S(r)

	def __matmul__(self, item):
		if (type(item) == dict): return S([i for i in self if all(self.get(i) in item[j] for j in item)])
		#if (len(item) == 1): return S([self[i].get(item[0]) for i in self])
		return S([self.get(i) for i in item])

	def __call__(self, *x):
		return S({i: self.get(i) for i in x})

	def __copy__(self):
		return Sdict(self.default_factory, self)

	__repr__ = dict.__repr__

	copy = __copy__

	def translate(self, table, copy=False, strict=True, keep=True):
		r = self.copy()
		for i in table:
			k, t = table[i] if (isinstance(table[i], (tuple, list))) else (table[i], lambda x: x)
			if (not strict and k not in r): continue
			if (keep and i not in r): r[i] = t((r.get if (copy) else r.pop)(k))
		return S(r)

	def with_(self, key, value=None):
		r = self.copy()
		r[key] = value
		return r

	_to_discard = set()

	def to_discard(self, x):
		self._to_discard.add(x)

	def discard(self):
		while (self._to_discard):
			try: self.pop(self._to_discard.pop())
			except: pass
Sdefaultdict = Sdict

class Slist(Stype, list):
	def __matmul__(self, item):
		if (type(item) == dict): return S([i for i in self if all(i.get(j) in (item[j]) for j in item)])
		r = S([[(i.get(j) if (hasattr(i, 'get')) else i[j]) for j in item] for i in self])
		return r.flatten() if (len(item) == 1 and not isiterable(item[0]) or type(item[0]) == str) else r

	def __getitem__(self, x):
		if (not isiterable(x)): return list.__getitem__(self, x)
		return S([i for i in self if (type(i) == dict and i.get(x[0]) in x[1:])])

	def __sub__(self, x):
		l = self.copy()
		for i in range(len(l)):
			if (all(l[i][j] == x[j] for j in x)): l.to_discard(i)
		l.discard()
		return l

	def copy(self):
		return S(list.copy(self))

	def group(self, n):
		return S([(*(j for j in i if (j is not None)),) for i in itertools.zip_longest(*[iter(self)]*n)])

	def flatten(self):
		return S([j for i in self for j in i])

	def strip(self, s=None):
		l = self.copy()
		l = l[0] if (len(l) == 1 and isiterable(l[0])) else l
		return S([i for i in l if (i if (s is None) else i != s)])

	def filter(self, t):
		return S([i for i in self if type(i) == t])

	_to_discard = set()

	def to_discard(self, x):
		self._to_discard.add(x)

	def discard(self):
		while (self._to_discard):
			try: self.pop(self._to_discard.pop())
			except: pass

class Stuple(Slist): pass # TODO

class Sint(Stype, int):
	def __len__(self):
		return Sint(math.log10(abs(self) or 10))

	def constrain(self, lb, ub):
		return S(constrain(self, lb, ub))

	def format(self, char=' '):
		return char.join(map(str().join, Slist(str(self)[::-1]).group(3)))[::-1]

class Sstr(Stype, str):
	def __and__(self, x):
		return Sstr().join(i for i in self if i in x)

	def fit(self, l, char='…'):
		return S(self if (len(self) <= l) else self[:l-1]+char)

	def join(self, l, last=None):
		l = tuple(map(str, l))
		return S((str.join(self, l[:-1])+(last or self)+l[-1]) if (len(l) > 1) else l[0] if (l) else '')

	def bool(self, minus_one=True):
		return bool(self) and self.casefold() not in ('0', 'false', 'no', 'нет', '-1'*(not minus_one))

	def indent(self, n=None, char=None, tab_width=8):
		if (not self): return self
		if (n is None): n = tab_width
		r, n = n < 0, abs(n)
		if (char is None): char = ('\t'*(n//tab_width)+' '*(n % tab_width)) if (not r) else ' '*n
		else: char *= n
		return char+('\n'+char).join(self.split('\n')) if (not r) else (char+'\n').join(self.split('\n'))+char

	def just(self, n, char=' ', j=None):
		if (j is None): j = '<>'[n>0]; n = abs(n)
		if (j == '.'): return self.center(n, char)
		elif (j == '<'): return self.ljust(n, char)
		elif (j == '>'): return self.rjust(n, char)
		else: raise ValueError

	def sjust(self, n, *args, **kwargs):
		return self.indent(n-len(self), *args, **kwargs)

	def wrap(self, w, char=' ', j='<'):
		r = self.split('\n')
		for ii, i in enumerate(r):
			if (len(i) > w): r.insert(ii+1, S(i[w:]).just(w, char, j=j)); r[ii] = r[ii][:w]
		return S('\n'.join(r))

	def filter(self, chars):
		return str().join(i for i in self if i in chars)

	def sub(self):
		return self.translate({
			ord('0'): '₀',
			ord('1'): '₁',
			ord('2'): '₂',
			ord('3'): '₃',
			ord('4'): '₄',
			ord('5'): '₅',
			ord('6'): '₆',
			ord('7'): '₇',
			ord('8'): '₈',
			ord('9'): '₉',
		})

	def super(self):
		return self.translate({
			ord('0'): '⁰',
			ord('1'): '¹',
			ord('2'): '²',
			ord('3'): '³',
			ord('4'): '⁴',
			ord('5'): '⁵',
			ord('6'): '⁶',
			ord('7'): '⁷',
			ord('8'): '⁸',
			ord('9'): '⁹',
			ord('.'): '·',
		})

def Sbool(x=bool(), *args, **kwargs): # No way to derive a class from bool
	x = S(x)
	return x.bool(*args, **kwargs) if (hasattr(x, 'bool')) else bool(x)

def dispatch_typecheck(o, t):
	return t is None or isinstance(o, typing_inspect.get_origin(t) or t) and (all(dispatch_typecheck(i, typing_inspect.get_args(t)) for i in o) if (isinstance(o, typing.Sequence) and not isinstance(o, str)) else True)
_overloaded_functions = Sdict(dict)
_overloaded_functions_retval = Sdict(dict)
_overloaded_functions_docstings = Sdict(dict)
def dispatch(f):
	fname = f.__qualname__
	fsig = inspect.signature(f)
	params_annotation = tuple((i[0], (None if (i[1].annotation is inspect._empty) else i[1].annotation, i[1].default is not inspect._empty, i[1].kind)) for i in fsig.parameters.items())
	_overloaded_functions[fname][params_annotation] = f
	_overloaded_functions_retval[fname][params_annotation] = fsig.return_annotation
	_overloaded_functions_docstings[fname][fsig] = f.__doc__
	#dplog([(dict(i), '—'*40) for i in _overloaded_functions[fname]], width=60) # XXX
	def overloaded(*args, **kwargs):
		args = list(args)
		for k, v in _overloaded_functions[fname].items():
			#dplog(k)
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

			r = v(*args, **kwargs); break
		else:
			if (() in _overloaded_functions[fname]): r = _overloaded_functions[fname][()](*args, **kwargs)
			else: raise TypeError(f"Parameters {(*map(type, args), *map(type, kwargs.values()))} doesn't match any of '{fname}' signatures:\n{S(overloaded_format_signatures(fname, f.__qualname__, sep=endl)).indent(2, char=' ')}")
		retval = _overloaded_functions_retval[fname][k]
		if (retval is not inspect._empty and not isinstance(r, retval)):
			raise TypeError(f"Return value of type {type(r)} doesn't match return annotation of appropriate '{fname}' signature")
		return r
	overloaded.__name__, overloaded.__qualname__, overloaded.__module__, overloaded.__doc__, overloaded.__signature__, overloaded.__code__ = f"Overloaded {f.__name__}", f.__qualname__, f.__module__, (_overloaded_functions_docstings[fname][()]+'\n\n' if (() in _overloaded_functions_docstings[fname]) else '')+overloaded_format_signatures(fname, f.__qualname__), ..., type(overloaded.__code__)(overloaded.__code__.co_argcount, overloaded.__code__.co_kwonlyargcount, overloaded.__code__.co_nlocals, overloaded.__code__.co_stacksize, overloaded.__code__.co_flags, overloaded.__code__.co_code, overloaded.__code__.co_consts, overloaded.__code__.co_names, overloaded.__code__.co_varnames, overloaded.__code__.co_filename, f"<overloaded '{f.__qualname__}'>", overloaded.__code__.co_firstlineno, overloaded.__code__.co_lnotab, overloaded.__code__.co_freevars, overloaded.__code__.co_cellvars) # it's soooooo long :D
	return overloaded
def dispatch_meta(f):
	_overloaded_functions_docstings[f.__qualname__][()] = f.__doc__
	return f
def overloaded_format_signatures(fname, qualname, sep='\n\n'): return sep.join(Sstr().join((qualname, sig, ':\n\b    '+doc if (doc) else '')) for sig, doc in _overloaded_functions_docstings[fname].items())

class cachedfunction:
	def __init__(self, f):
		self.f = f
		self._cached = dict()

	def __call__(self, *args, **kwargs):
		k = (args, tuple(sorted(kwargs.items())))
		if (k not in self._cached):
			self._cached[k] = self.f(*args, **kwargs)
		return self._cached[k]

# Здесь должен быть текст, но прибежал Лукин и его спёр
#  25.03.19, 3:18

class cachedproperty:
	class _empty: __slots__ = ()
	_empty = _empty()

	def __init__(self, f):
		self.f = f
		self._cached = self._empty

	def __get__(self, obj, objcls):
		if (self._cached is self._empty):
			self._cached = self.f(obj)
		return self._cached

def spreadargs(f, okwargs, *args, **akwargs):
	fsig = inspect.signature(f)
	kwargs = S(okwargs) & akwargs
	try: kwnames = (*(i[0] for i in fsig.parameters.items() if (assert_(i[1].kind != inspect.Parameter.VAR_KEYWORD) and i[1].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))),)
	except AssertionError: kwnames = kwargs.keys()
	for i in kwnames:
		try: del okwargs[i]
		except KeyError: pass
	return f(*args, **kwargs)

logcolor = ('\033[94m', '\033[92m', '\033[93m', '\033[91m', '\033[95m')
noesc = re.compile(r'\033\[?[0-?]*[ -/]*[@-~]?')
logfile = None
logoutput = sys.stderr
loglock = [(0,)]
_logged_utils_start = None
def log(l=None, *x, sep=' ', end='\n', ll=None, raw=False, tm=None, format=False, width=80, unlock=0, nolog=False): # TODO: finally rewrite me as class pls
	""" Log anything. Print (formatted with datetime) to stderr and logfile (if set). Should be compatible with builtins.print().
	Parameters:
		l (optional): level, must be >= global loglevel to print to stderr rather than only to logfile (or /dev/null).
		*x: print-like args, what to log.
		sep: print-like separator.
		end: print-like line end.
		raw: flag, do not print datetime/loglevel prefix if set.
		tm: specify datetime, time.time() if not set.
		unlock: [undocumented]
		nolog: flag, force suppress printing to stderr.
	"""
	global loglock, _logged_utils_start
	if (isinstance(_logged_utils_start, tuple)): _logged_utils_start, _logstateargs = True, _logged_utils_start; logstart('Utils'); logstate(*_logstateargs)
	_l, _x = map(copy.copy, (l, x))
	if (l is None): l = ''
	if (type(l) is not int): l, x = None, (l, *x)
	if (x == ()): l, x = 0, (l,)
	x = 'plog():\n'*bool(format)+sep.join(map((lambda x: pformat(x, width=width)) if (format) else str, x))
	clearstr = noesc.sub('', str(x))
	if (tm is None): tm = time.localtime()
	if (not unlock and loglock[-1][0]): loglock[-1][1].append(((_l, *_x), dict(sep=sep, end=end, raw=raw, tm=tm, nolog=nolog))); return clearstr
	try: lc = logcolor[l]
	except: lc = ''
	if (ll is None): ll = f' [\033[1m{lc}LV{l}\033[0;96m]' if (l is not None) else ''
	logstr = f"\033[K\033[96m[{time.strftime('%x %X', tm)}]\033[0;96m{ll}\033[0;1m{' '+x if (x != '') else ''}\033[0m{end}" if (not raw) else str(x)+end
	if (logfile and not nolog): logfile.write(logstr)
	if (l is None or l <= loglevel): logoutput.write(logstr); logoutput.flush()
	if (unlock and loglock[-1][0] == unlock):
		_loglock = loglock.pop()
		for i in _loglock[1]: log(*i[0], **i[1], unlock=_loglock[0])
	return clearstr
def plog(*args, **kwargs): parseargs(kwargs, format=True); return log(*args, **kwargs)
def dlog(*args, **kwargs): parseargs(kwargs, ll=' \033[95m[\033[1mDEBUG\033[0;95m]\033[0;96m'); return log(*args, **kwargs)
def dplog(*args, **kwargs): parseargs(kwargs, format=True); return dlog(*args, **kwargs)
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
	log(state+(': '+str(x))*bool(str(x))+'\033[0m', raw=True, unlock=loglock[-1][0])
def logstarted(x=''): """ if (__name__ == '__main__'): logstart(); main() """; logstate('\033[94mstarted', x)
def logimported(x=''): """ if (__name__ != '__main__'): logimported() """; logstate('\033[96mimported', x)
def logok(x=''): logstate('\033[92mok', x)
def logex(x=''): logstate('\033[91merror', unraise(x))
def logwarn(x=''): logstate('\033[93mwarning', x)
def setlogoutput(f): global logoutput; logoutput = f
def setlogfile(f): global logfile; logfile = open(f, 'a')
def setloglevel(l): global loglevel; loglevel = l
def locklog(): loglock.append((loglock[-1][0]+1, list()))
def unlocklog():
	while (loglock[-1][0]): logdumb(unlock=loglock[-1][0])
def setutilsnologimport(): global _logged_utils_start; _logged_utils_start = True

_exc_handlers = set()
def register_exc_handler(f): _exc_handlers.add(f)
_logged_exceptions = set()
@dispatch
def exception(ex: BaseException, once=False, nolog=False):
	""" Log an exception.
	ex: exception to log.
	nolog: no not call exc_handlers.
	"""
	ex = unraise(ex)
	if (isinstance(ex, NonLoggingException)): return
	if (once):
		if (repr(ex) in _logged_exceptions): return
		_logged_exceptions.add(repr(ex))
	exc = repr(ex).split('(')[0]
	e = log(('\033[91mCaught ' if (not isinstance(ex, Warning)) else '\033[93m' if ('warning' in ex.__class__.__name__.casefold()) else '\033[91m')+f"{exc}{(' on line '+'→'.join(map(lambda x: str(x[1]), traceback.walk_tb(ex.__traceback__)))).rstrip(' on line')}\033[0m{(': '+str(ex))*bool(str(ex))}")
	if (nolog): return
	for i in _exc_handlers:
		try: i(e, ex)
		except Exception: pass
logexception = exception

@dispatch
def raise_(ex: BaseException): raise ex
@dispatch
def unraise(ex: (BaseException, type)):
	if (isinstance(ex, BaseException)): return ex
	elif (issubclass(ex, BaseException)): return ex()
	else: raise TypeError()
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

	def __init__(self, file=None):
		self.setfile(file)
		self.fields = dict()
		self.setnolog(False)
		self.setbackup(True)

	def setfile(self, file):
		self.file = file
		if (file is None): return False
		if ('/' not in file): file = os.path.dirname(os.path.realpath(sys.argv[0]))+'/'+file
		try: self.file = open(file, 'r+b')
		except FileNotFoundError: self.file = open(file, 'w+b')
		return True

	def setnolog(self, nolog=True):
		self.nolog = bool(nolog)

	def setbackup(self, backup):
		self.backup = bool(backup)

	def register(self, *fields):
		globals = inspect.stack()[1][0].f_globals
		for field in fields: self.fields[field] = globals

	def load(self, nolog=None):
		nolog = (self.nolog if (nolog is None) else nolog)
		if (not self.file): return
		if (not nolog): logstart('Loading database')
		db = dict()
		try: db = dill.load(self.file);
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
			except: pass
			try: shutil.copyfile(self.file.name, f"backup/{self.file.name if (hasattr(self.file, 'name')) else ''}_{time.time()}.db")
			except: pass
		try: dill.dump(db or {field: self.fields[field][field] for field in self.fields}, self.file)
		except Exception as ex:
			if (not nolog): logex(ex)
		else:
			if (not nolog): logok()
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
		eta = ' '.join(reversed(tuple(str(int(i))+'smhd'[ii] for ii, i in enumerate(S([eta%60, eta//60%60, eta//60//60%24, eta//60//60//24]).strip())))) or '?'
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
		self.p = p

	def __del__(self):
		for i in self.p: i.printed = False
		sys.stderr.write('\033[J')
		sys.stderr.flush()

	@dispatch
	def __init__(self, n: int):
		self.p = (*(Progress() for _ in range(n)),)

	def print(self, *cvs, width=None):
		self.p[0].print(cvs[0], width=width)
		for p, cv in zip(self.p[1:], cvs[1:]):
			sys.stderr.write('\n')
			p.print(cv, width=width)
		sys.stderr.write(f"\033[{len(self.p)-1}A")
		sys.stderr.flush()

	def done(self, width=None):
		self.print(*(i.mv for i in self.p), width=width)

class ThreadedProgressPool(ProgressPool, threading.Thread):
	def __init__(self, *args, width=None, delay=0.01, **kwargs):
		threading.Thread.__init__(self, daemon=True)
		super().__init__(*args, **kwargs)
		self.width, self.delay = width, delay
		self.stopped = bool()
		self.cvs = [int()]*len(self.p)

	def run(self):
		while (not self.stopped):
			locklog()
			self.print(*self.cvs, width=self.width)
			unlocklog()
			time.sleep(self.delay)

	def stop(self):
		self.stopped = True
		self.join()

def testprogress(n=1000, sleep=0.002):
	p = Progress(n)
	for i in range(n+1):
		p.print(i)
		time.sleep(sleep)

def progrange(start, stop=None, step=1):
	if (stop is None): start, stop = 0, start
	p = Progress(stop-start-1)
	for i in range(start, stop, step):
		p.print(i)
		yield i

def progiter(iterable):
	try: p = Progress(len(iterable))
	except TypeError: yield from iterable; return
	for ii, i in enumerate(iterable):
		p.print(ii+1)
		yield i

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

def decline(n, w, prefix=('',)*3, format=False, show_one=True):
	q = abs(n) % 10
	if (5 <= abs(n) <= 20): q = 0
	if (q == 1): r, p = w[0], prefix[0]
	elif (2 <= q <= 4): r, p = w[1], prefix[1]
	else: r, p = w[2], prefix[2]
	return f"{(p+' ') if (p) else ''}{str(S(n).format(format if (format is not True) else ' ') if (format) else n)+' ' if (n != 1 or show_one) else ''}{r}"
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
	while (q.size()): yield q.get()

function = type(lambda: None)

def pm(x): return 1 if (x) else -1
def constrain(x, lb, ub):
	r = min(ub, max(lb, x))
	assert lb <= r <= ub
	return r
def prod(x):
	r = 1
	for i in x: r *= i
	return r

global_lambdas = list() # dunno why
def global_lambda(l): global_lambdas.append(l); return global_lambdas[-1]

class lc:
	def __init__(self, lc):
		self.lc = lc

	def __enter__(self, *args, **kwargs):
		self.pl = locale.setlocale(locale.LC_ALL)
		locale.setlocale(locale.LC_ALL, self.lc)

	def __exit__(self, *args, **kwargs):
		locale.setlocale(locale.LC_ALL, self.pl)

class ll:
	def __init__(self, ll):
		self.ll = ll

	def __enter__(self, *args, **kwargs):
		self.pl = loglevel
		setloglevel(self.ll)

	def __exit__(self, *args, **kwargs):
		setloglevel(self.pl)

class classproperty:
	def __init__(self, f):
		self.f = f

	def __get__(self, obj, owner):
		return self.f(owner)

def preeval(f):
        r = f()
        return lambda: r

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

class WTFException(Exception): pass
class TODO(NotImplementedError): pass
class TEST(BaseException): pass
class NonLoggingException(Exception): pass

def setonsignals(f): signal.signal(signal.SIGINT, f); signal.signal(signal.SIGTERM, f)

def exit(c=None, code=None, raw=False, nolog=False):
	sys.stderr.write('\r')
	unlocklog()
	db.save(nolog=True)
	if (not nolog): log(f'{c}' if (raw) else f'Exit: {c}' if (c and type(c) == str or hasattr(c, 'args') and c.args) else 'Exit.')
	try: sys.exit(int(bool(c)) if (code is not None) else code)
	finally:
		if (not nolog): log(raw=True)

logstart('Utils')
if (__name__ == '__main__'):
	testprogress()
	log('\033[0mWhy\033[0;2m are u trying to run me?! It \033[0;93mtickles\033[0;2m!..\033[0m', raw=True)
else: logimported()

# by Sdore, 2019
