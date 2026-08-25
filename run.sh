#!/bin/bash
# Launch the demo (Linux/macOS/Git Bash). Assumes requirements installed + checkpoints fetched.
cd "$(dirname "$0")"

# -X utf8 matters on Windows: the picker labels contain '·', and a cp1252 console raises
# UnicodeEncodeError without it. -u keeps the startup line from sitting in the stdout buffer.
for py in .venv/Scripts/python.exe .venv/bin/python venv/Scripts/python.exe venv/bin/python; do
  if [ -x "$py" ]; then exec "$py" -X utf8 -u app.py "$@"; fi
done
exec python -X utf8 -u app.py "$@"
