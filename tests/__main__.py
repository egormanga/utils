#!/usr/bin/env python3
# Utils lib tests

from utils import *; logstart('Utils tests')

def main():
	exec('from . import '+', '.join(i.rpartition('.')[0] for i in os.listdir(os.path.dirname(__file__)) if i.endswith('.py')))
	log('Utils tests ok.')

if (__name__ == '__main__'): logstarted(); exit(main())

# by Sdore, 2019-25
#   www.sdore.me
