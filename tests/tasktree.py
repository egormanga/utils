#!/usr/bin/env python3
# Utils lib TaskTree test

import time
from utils.nolog import log, TaskTree, NodesTree

def check(v, s):
	print(v)
	assert (v == s)

check(NodesTree({
	1: {},
	2: {
		3: {},
		4: {
			5: {},
			6: {},
		},
		'asd': {
			7: {},
		},
		'empty': {},
		'recursive': {
			1: {
				2: {
					3: {},
					4: {},
					5: {},
				},
				6: {},
			},
			7: {
				8: {},
			},
		},
	},
	'last': {},
}).format(), """\
┌── 1
├── 2
│   ├── 3
│   ├── 4
│   │   ├── 5
│   │   └── 6
│   ├── asd
│   │   └── 7
│   ├── empty
│   └── recursive
│       ├── 1
│       │   ├── 2
│       │   │   ├── 3
│       │   │   ├── 4
│       │   │   └── 5
│       │   └── 6
│       └── 7
│           └── 8
└── last""")

print()

check(NodesTree({
	1: {2: 3},
}).format(), """\
╾── 1
    └── 2
        └── 3""")

print()

tasks = [
	TaskTree.Task('Task 1', [
		TaskTree.Task('0%', []),
		TaskTree.Task('0%', []),
	]),
	TaskTree.Task('Task 2', [
		TaskTree.Task('0%', []),
		TaskTree.Task('0%', []),
	]),
]
tt = TaskTree(tasks)
for i in tasks:
	for t in i.subtasks:
		for j in range(10):
			t.title = f"{(j+1)*10}%"
			tt.print()
			time.sleep(0.05)
		t.state = True
	i.state = True
	tt.print()
del tt

log('TaskTree test ok\n')

# by Sdore, 2019-25
#   www.sdore.me
