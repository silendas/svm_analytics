import sys
import os

INTERP = os.path.expanduser("/home/USERNAME/YOUR_DOMAIN_NAME/venv/bin/python")
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

sys.path.append(os.getcwd())

from app import app as application 