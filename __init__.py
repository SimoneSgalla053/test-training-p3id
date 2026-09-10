"""Legacy package exports; CLI checkouts can also be imported by test discovery."""

if __package__:
    from .utils import *
