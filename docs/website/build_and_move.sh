#!/bin/bash

make html

if [ -d "build/html" ]; then
    cp -R build/html/* ..

    rm -rf build/html
else
    echo "Error: build/html folder not found."
fi
