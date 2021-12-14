
import os
import psutil, pynvml

from .tools import Singleton

G = 1024*1024*1024

# !warning TODO

class ResourceManager(Singleton):
    # Minimal free memory(MiB) for new process to run and minimal cpu free percent
    def __init__(self, mem_limit=0, cpu_limit=0, gpu_limit=0, max_instances=9999):
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.gpu_limit = gpu_limit
        self.max_instances = max_instances
        
        print('Find CPU count:',psutil.cpu_count())
        info = psutil.virtual_memory()
        print('Find total memory:', round(info.total/G), 'G')
        
        pynvml.nvmlInit()
        self.gpu_num = pynvml.nvmlDeviceGetCount()
        self.handles = []
        for gpu_id in range(self.gpu_num):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            print("Find GPU", gpu_id, ":", pynvml.nvmlDeviceGetName(handle).decode('utf-8'))
            self.handles.append(handle)
        
    def __del__(self):
        pynvml.nvmlShutdown()
        
    def get_gpu_access(self):
        free_mems = []
        for handle in self.handles:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # info.total,info.free,info.used (B)
            free_mems.append(info.free)
        max_free_mem = max(free_mems)
        gpu_id = free_mems.index(max_free_mem)
        if max_free_mem/G < self.gpu_limit:
            # no enough memory to use
            return -1
        else:
            # return gpu id with maximal memory
            return gpu_id
        
    def get_memory_access(self):
        info = psutil.virtual_memory()
        if info.available/G < self.mem_limit:
            return False
        return True
    
    def get_cpu_access(self):
        if 1. - psutil.cpu_percent(interval=0.5)/100. < self.cpu_limit:
            return False
        return True
    
    def report(self):
        print('**************** REPORT ****************')
        info = psutil.virtual_memory()
        print('Memory usage :', round(info.used/G, 2), 'G /',
              round(info.total/G, 2),'G')
        print('CPU usage:',psutil.cpu_percent(interval=0.5), '%')
        for i in range(len(self.handles)):
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handles[i])
            print('GPU', i, 'usage:', round(info.used/G, 2), 'G /',
                  round(info.total/G, 2),'G')
        print('****************************************')

    def cpu_usage(self):
        pass

    def memory_usage(self, pid=None):
        info = psutil.virtual_memory()
        total = info.total / G
        if pid is not None:
            current = psutil.Process(os.getpid()).memory_info().rss / G
            print('memery use (current): %.4f GB / %.4f GB' % (current, total))
        else:
            print('memery use (total): %.4f GB / %.4f GB' % (info.used / G, total))
        return
    
    def gpu_memory_usage(self, pid=None):
        for i, handle in enumerate(self.handles):
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print('GPU', i, 'usage:', round(info.used/G, 2), 'G /',
                  round(info.total/G, 2),'G')

