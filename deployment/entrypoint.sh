#!/bin/bash 

echo 'Running unittests...'
python -m pytest -s

if [ $? -ne 0 ];
    then echo 'Unittests failed'
    exit 1;
fi

echo 'Running ASGI Server...'
uvicorn rest.settings --port 8080 

if [ $? -ne 0 ];
    then echo "Failed to start ASGI Server"
    exit 1;
fi