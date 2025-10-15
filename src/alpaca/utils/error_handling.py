# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""


class TimeoutException(Exception):
    """Custom exception to be raised on timeout."""


def timeout_handler(signum, frame):
    """Handler function to raise a TimeoutException."""
    raise TimeoutException("Model data buildup exceeded the time limit.")
