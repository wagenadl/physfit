#!/bin/sh

python3 -m build

twine upload dist/physfit-0.9.0-py3-none-any.whl
