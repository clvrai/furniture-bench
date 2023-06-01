#!/bin/bash

# Check if SERVER_DOCKER environment variable is set, without checking not set.
if [ -n "$SERVER_DOCKER" ]; then
    echo "Using $SERVER_DOCKER"
else
    built=false
    pulled=false
    if [ "$1" = "--built" ]; then
        built=true
    elif [ "$1" = "--pulled" ]; then
        pulled=true
    elif [ -z "$1" ]; then
        echo "No first argument provided"
        exit 1
    else
        echo "Unknown first argument: $1"
        exit 1
    fi

    if [ "$built" = true ]; then
        SERVER_DOCKER=server
    elif [ "$pulled" = true ]; then
        SERVER_DOCKER=furniturebench/server:latest
    else
        echo "Unknown first argument: $1"
        exit 1
    fi
fi

# Check if environment variable FURNITURE_BENCH is set
if [ -z "$FURNITURE_BENCH" ]; then
    echo "FURNITURE_BENCH is not set"
    exit 1
fi

echo -e "Environment Vairables"
echo "---------------------"
echo "SERVER_DOCKER: $SERVER_DOCKER"
echo "FURNITURE_BENCH: $FURNITURE_BENCH"


docker run -it --rm --network=host --privileged -v $FURNITURE_BENCH:/furniture-bench $SERVER_DOCKER /bin/bash
