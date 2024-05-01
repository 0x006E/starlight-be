import asyncio
import itertools
import threading

__all__ = ["EventLoopThread", "get_event_loop", "stop_event_loop", "run_coroutine"]


class EventLoopThread(threading.Thread):
    loop = None
    _count = itertools.count(0)

    def __init__(self):
        self.started = threading.Event()
        name = f"{type(self).__name__}-{next(self._count)}"
        super().__init__(name=name, daemon=True)

    def __repr__(self):
        loop, r, c, d = self.loop, False, True, False
        if loop is not None:
            r, c, d = loop.is_running(), loop.is_closed(), loop.get_debug()
        return (
            f"<{type(self).__name__} {self.name} id={self.ident} "
            f"running={r} closed={c} debug={d}>"
        )

    def run(self):
        self.loop = loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.call_later(0, self.started.set)

        try:
            loop.run_forever()
        finally:
            try:
                shutdown_asyncgens = loop.shutdown_asyncgens()
            except AttributeError:
                pass
            else:
                loop.run_until_complete(shutdown_asyncgens)
            try:
                shutdown_executor = loop.shutdown_default_executor()
            except AttributeError:
                pass
            else:
                loop.run_until_complete(shutdown_executor)
            asyncio.set_event_loop(None)
            loop.close()

    def stop(self):
        loop, self.loop = self.loop, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        self.join()


_lock = threading.Lock()
_loop_thread = None


def get_event_loop():
    global _loop_thread

    if _loop_thread is None:
        with _lock:
            if _loop_thread is None:
                _loop_thread = EventLoopThread()
                _loop_thread.start()
                # give the thread up to a second to produce a loop
                _loop_thread.started.wait(1)

    return _loop_thread.loop


def stop_event_loop():
    global _loop_thread
    with _lock:
        if _loop_thread is not None:
            _loop_thread.stop()
            _loop_thread = None


def run_coroutine(coro):
    """Run the coroutine in the event loop running in a separate thread

    Returns a Future, call Future.result() to get the output

    """
    return asyncio.run_coroutine_threadsafe(coro, get_event_loop())
