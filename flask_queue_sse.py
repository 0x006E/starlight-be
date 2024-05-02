#  A simple implementation of Server-Sent Events for Flask
#  that doesn't require Redis pub/sub.
#  Created On 21 November 2022
#

from re import search
from json import dumps
from queue import Queue, Full
from flask import Response


def format_sse(data: str, event=None) -> str:
    msg = f"data: {data}\n\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    return msg


class ServerSentEvents:
    """A simple implementation of Server-Sent Events for Flask."""

    msg_id: int = 0
    listener: Queue = Queue(1000)

    def response(self):
        """Returns a response which can be passed to Flask server."""

        def stream():
            has_finished = False

            while has_finished == False:
                msg = self.listener.get()
                yield msg

                if search("event: end", msg) or search("event: error", msg):
                    has_finished = True

        return Response(stream(), mimetype="text/event-stream")

    def send(self, payload: dict = None, event: str = None):
        """Sends a new event to the opened channel."""

        self.msg_id = self.msg_id + 1

        msg_str = dumps(payload) if payload else "{}"
        msg = format_sse(msg_str, event=event)

        try:
            self.listener.put_nowait(msg)
        except Full:
            print("QueueFull")

        return self
