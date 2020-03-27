import os
from multiprocessing import Process

def _worker(logdir):
    os.system(f'tensorboard --logdir={logdir}')

def launch_tensorboard(logdir):
    process = Process(target=_worker, args=[logdir], daemon=True)
    process.start()
