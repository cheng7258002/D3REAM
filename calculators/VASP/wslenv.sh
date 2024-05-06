#!/bin/bash

export INTEL=/mnt/e/WSL_program/intel
export PATH=$INTEL/bin:$PATH
export LD_LIBRARY_PATH=$INTEL/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$INTEL/lib:LIBRARY_PATH
COMPILERVARS_PLATFORM=linux
COMPILERVARS_ARCHITECTURE=intel64
source $INTEL/bin/ifortvars.sh
source $INTEL/bin/iccvars.sh
source $INTEL/bin/compilervars.sh

export PATH="$PATH:/mnt/e/VASP/vasp.5.4.4/vasp.5.4.4/bin"

export PATH="/mnt/e/VASP/vaspkit.1.2.5/bin:$PATH"
