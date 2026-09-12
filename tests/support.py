"""Constants shared by the fixtures and the test modules."""

import shutil

from hermes_nari.api import Timeouts

REAL_FFMPEG = shutil.which("ffmpeg")
FAST_TIMEOUTS = Timeouts(tts=10, voices=5, ws_open=5, ws_configure=5, ws_result=10)
