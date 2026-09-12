"""Fake ffmpeg used by the tests (the conftest prepends a Python shebang and puts it on PATH).

It records every invocation as a JSON line in $FAKE_FFMPEG_LOG and produces deterministic
output: decoding (``-f s16le`` to stdout) echoes the input file's bytes; encoding writes
``FAKE:<format>:`` followed by the input bytes. FAKE_FFMPEG_FAIL forces a failure.
"""

import json
import os
import sys


def main(argv):
    log = os.environ.get("FAKE_FFMPEG_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(argv) + "\n")
    if os.environ.get("FAKE_FFMPEG_FAIL"):
        sys.stderr.write("fake ffmpeg: forced failure\n")
        return 3
    source = None
    fmt = None
    previous = None
    for arg in argv:
        if previous == "-i":
            source = arg
        elif previous == "-f":
            fmt = arg
        previous = arg
    output = argv[-1] if argv else "-"
    if source in ("pipe:0", "-"):
        data = sys.stdin.buffer.read()
    else:
        with open(source, "rb") as handle:
            data = handle.read()
    payload = data if fmt == "s16le" else f"FAKE:{fmt}:".encode() + data
    if output in ("-", "pipe:1"):
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    else:
        with open(output, "wb") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
