import multiprocessing as mp
import queue
import threading


def buffered_generator_process(generator, buffer_size=2):
    if buffer_size < 2:
        raise RuntimeError("Minimum buffer size is 2")
    buffer = mp.Queue(maxsize=buffer_size - 1)

    def _buffered_generation_process(generator, buffer):
        for data in generator:
            buffer.put(data, block=True)
        buffer.put(None)
        buffer.close()

    process = mp.Process(target=_buffered_generation_process,
                         args=(generator, buffer))
    process.start()
    for data in iter(buffer.get, None):
        yield data


def buffered_generator_thread(generator, buffer_size=2):
    if buffer_size < 2:
        raise RuntimeError("Minimum buffer size is 2")
    buffer = queue.Queue(maxsize=buffer_size - 1)

    def _buffered_generation_thread(generator, buffer):
        for data in generator:
            buffer.put(data, block=True)
        buffer.put(None)

    thread = threading.Thread(target=_buffered_generation_thread,
                              args=(generator, buffer))
    thread.daemon = True
    thread.start()
    for data in iter(buffer.get, None):
        yield data
