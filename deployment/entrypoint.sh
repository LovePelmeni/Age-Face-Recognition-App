#!/bin/bash 

echo "Running autoformatters.."

autopep8 . --in-place --recursive

if [ $? -ne 0 ];
    then echo "Autoformatting failed"
    exit 1;
fi;

echo "Running linters.."

flake8 .

if [ $? -ne 0 ]; 
    then echo "Code does not have appropriate format"
    exit 1;
fi

echo 'Running unittests...'
python -m pytest -s

if [ $? -ne 0 ];
    then echo 'Unittests failed'
    exit 1;
fi

echo 'Running ASGI Server...'
uvicorn rest.settings --port 8080 --num-workers 15

if [ $? -ne 0 ];
    then echo "Failed to start ASGI Server"
    exit 1;
fi