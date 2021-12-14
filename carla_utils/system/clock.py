
import time


class Clock(object):
    def __init__(self, frequency):
        self.frequency = float(frequency)
        self.dt = 1 / self.frequency
        self._tick_begin = None

    def tick_begin(self):
        self._tick_begin = time.time()
        return self._tick_begin
    def tick_end(self):
        _tick_end = time.time()
        sleep_time = self.dt - _tick_end + self._tick_begin
        time.sleep( max(0, sleep_time) )
        return _tick_end
