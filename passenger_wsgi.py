import os
import sys
import resource

# Set RLIMIT_NPROC
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (2000, 2000))
except:
    pass

# Set environment variables untuk OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Set Python path
INTERP = os.path.expanduser("~/virtualenv/svm.chasouluix.my.id/3.9/bin/python")
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

cwd = os.getcwd()
sys.path.append(cwd)

# Import aplikasi
from app import application