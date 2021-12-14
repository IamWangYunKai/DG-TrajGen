from multiprocessing import Lock
from multiprocessing.managers import BaseManager

class Variable(object):
    def __init__(self):
        self.data_forward, self.data_backward = None, None
        self.lock_forward, self.lock_backward = Lock(), Lock()
    
    def write_forward(self, data):
        with self.lock_forward:
            self.data_forward = data
    def read_forward(self):
        return self.data_forward

    def write_backward(self, data):
        with self.lock_backward:
            self.data_backward = data
    def read_backward(self):
        return self.data_backward


BaseManager.register('Variable', Variable)
manager = BaseManager()

def SharedVariable():
    manager.start()
    return manager.Variable()