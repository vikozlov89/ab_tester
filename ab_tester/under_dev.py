import numpy as np
from queue import Queue
from threading import Thread




class BootstrapQueueFiller(Thread):

    def __init__(self, queue: Queue, n_boots: int, n_threads: int):
        super().__init__()
        self.queue = queue
        self.n_boots = n_boots
        self.n_threads = n_threads

    def run(self) -> None:
        step = self.n_boots // self.n_threads
        start_index = 0
        end_index = step

        while start_index < self.n_boots:
            self.queue.put(min(step, self.n_boots - start_index))
            start_index = end_index
            end_index += step

        self.queue.task_done()
        return None

class BootstrapWorker(Thread):

    def __init__(self, sample, simulator, queue, simulation_params, result_queue):
        super().__init__()
        self.sample = sample
        self.simulator = simulator
        self.daemon = True
        self.queue = queue
        self.simulation_params = simulation_params
        self.result_queue = result_queue

    def run(self) -> None:

        while True:
            try:

                # print('Starting Boot Worker...')
                n_boots = self.queue.get()

                if 'n_boots' in self.simulation_params:
                    self.simulation_params['n_boots'] = n_boots
                # print(self.queue.qsize())
                # print(f'Performing {n_boots} boots')
                res = self.simulator.boot(sample=self.sample, **self.simulation_params)
                self.result_queue.put(res)
            finally:
                pass
                # print('Task is done...')
            if self.queue.empty():
                break

        self.queue.task_done()
        self.result_queue.task_done()
        # print('finishing...')

class ResultQueueConverter(Thread):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue
        self.result = []

    def run(self) -> None:
        while True:
            try:
                tmp = self.queue.get()
                self.result.append(tmp)
            finally:
                pass
            if self.queue.empty():
                break
        self.queue.task_done()


def multithread_bootstrap(sample, simulator, n_threads, simulation_params):

    queue = Queue()
    n_boots = simulation_params.get('n_boots')
    queue_filler = BootstrapQueueFiller(queue, n_boots, n_threads)
    result_queue = Queue()
    for w in range(n_threads):
        worker = BootstrapWorker(sample, simulator, queue, simulation_params, result_queue)
        worker.start()
    queue_filler.start()
    queue.join()
    conv = ResultQueueConverter(result_queue)
    conv.start()
    result_queue.join()
    return np.hstack(conv.result)