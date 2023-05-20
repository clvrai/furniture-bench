#!/bin/bash

make html

if [ -d "build/html" ]; then
    mkdir -p docs
    cp -R build/html/* docs/
    rm -rf build
else
    echo "Error: build/html folder not found."
fi
