
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from .never_complain_queue import NeverComplainQueue


# create manager that knows how to create and manage LifoQueues
class NCQManager(BaseManager):
    pass

def NCQPipe(maxsize):
    NCQManager.register('NeverComplainQueue', NeverComplainQueue)
    manager = NCQManager()
    manager.start()
    a, b = manager.NeverComplainQueue(maxsize), manager.NeverComplainQueue(maxsize)
    return Connection(a, b), Connection(b, a)



class Connection(object):
    def __init__(self, _in, _out):
        self._out = _out
        self._in = _in
        self.send = self.send_bytes = _out.put
        self.recv = self.recv_bytes = _in.get


    def poll(self, timeout=0.0):
        if self._in.qsize() > 0:
            return True
        if timeout <= 0.0:
            return False
        with self._in.not_empty:
            self._in.not_empty.wait(timeout)
        return self._in.qsize() > 0

    @property
    def send_size(self):
        return self._out.size()
    @property
    def recv_size(self):
        return self._in.size()

    # def send(self, item):
    #     self._latest_time_stamp = item[0]
    #     self._latest_item = item
    #     self._send(item)

    # def recv(self):
    #     item = self._recv()
    #     if item is None and self._latest_item is None:
    #         return None
    #     elif item is None and self._latest_item is not None:
    #         return self._latest_item
    #     else:
    #         if item[0] < self._latest_time_stamp:
    #             return self._latest_item
    #         else:
    #             return item
    
    

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()