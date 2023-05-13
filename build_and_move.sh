#!/bin/bash

make html

if [ -d "build/html" ]; then
    cp -R build/html/* docs/
    rm -rf build
else
    echo "Error: build/html folder not found."
fi
