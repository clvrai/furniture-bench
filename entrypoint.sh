#!/bin/bash
/opt/conda/envs/${VENV_NAME}/bin/pip install -e /furniture-bench

if [ -d "/isaacgym" ]; then
    /opt/conda/envs/${VENV_NAME}/bin/pip install -e /isaacgym/python
fi

exec /bin/bash "$@"