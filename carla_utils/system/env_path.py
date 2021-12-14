
import sys

python2_path = []
for p in sys.path:
    if 'python2' in p:
        python2_path.append(p)

def remove_python2_path():
    for p in python2_path:
        sys.path.remove(p)

def append_python2_path():
    sys.path.extend(python2_path)


def append(path):
	sys.path.append(path)

def insert(index, path):
    sys.path.insert(index, path)

def print():
    print(sys.path)