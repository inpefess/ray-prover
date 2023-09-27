#!/bin/bash

set -e
PACKAGE_NAME=ray_prover
cd doc
make clean html coverage
cat _build/coverage/python.txt
cd ..
pydocstyle ${PACKAGE_NAME}
flake8 ${PACKAGE_NAME}
pylint ${PACKAGE_NAME}
mypy ${PACKAGE_NAME}
find ${PACKAGE_NAME} -name "*.py" | xargs -I {} pyupgrade --py38-plus {}
pyroma -n 10 .
bandit -r ${PACKAGE_NAME}
pytest --cov-report term-missing
scc --no-cocomo --by-file -i py ${PACKAGE_NAME}
