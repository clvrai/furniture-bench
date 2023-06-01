#!/bin/bash

# Set default options
gpu=false
cpu=false
sim_gpu=false

# Parse the first argument.
case $1 in
    --gpu)
    gpu=true
    ;;
    --cpu)
    cpu=true
    ;;
    --sim-gpu)
    sim_gpu=true
    ;;
    *)    # unknown option
    echo "Unknown option: $1"
    exit 1
    ;;
esac

# Check if CLIENT_DOCKER environment variable is set, without checking not set.
if [ -n "$CLIENT_DOCKER" ]; then
    echo "Using $CLIENT_DOCKER"
else
    # Use second argument to determine which CLIENT_DOCKER to use.
    built=false
    pulled=false
    if [ "$2" = "--built" ]; then
        built=true
    elif [ "$2" = "--pulled" ]; then
        pulled=true
    elif [ -z "$2" ]; then
        echo "No second argument provided"
        exit 1
    else
        echo "Unknown second argument: $2"
        exit 1
    fi

    if [ "$built" = true ]; then
        if [ "$gpu" = true ] || [ "$sim_gpu" = true ]; then
            CLIENT_DOCKER="client-gpu"
        elif [ "$cpu" = true ]; then
            CLIENT_DOCKER="client"
        else
            echo "No option provided"
            exit 1
        fi
    elif [ "$pulled" = true ]; then
        if [ "$gpu" = true ] || [ "$sim_gpu" = true ]; then
            CLIENT_DOCKER="furniturebench/client-gpu:latest"
        elif [ "$cpu" = true ]; then
            CLIENT_DOCKER="furniturebench/client:latest"
        else
            echo "No option provided"
            exit 1
        fi
    else
        echo "Unknown second argument: $2"
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
echo "CLIENT_DOCKER: $CLIENT_DOCKER"
echo "FURNITURE_BENCH: $FURNITURE_BENCH"
echo "HOST_DATA_MOUNT: $HOST_DATA_MOUNT"
echo "CONTAINER_DATA_MOUNT: $CONTAINER_DATA_MOUNT"
echo "ISAAC_GYM_PATH: $ISAAC_GYM_PATH"

# Allow docker to connect to X server
xhost +

if [ "$gpu" = true ]; then
    # Run nvidia-docker command with GPU option
    docker run --network host -it --privileged \
    -v $FURNITURE_BENCH:/furniture-bench \
    --gpus=all --rm --ipc=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOST_DATA_MOUNT:$CONTAINER_DATA_MOUNT \
    --env="QT_X11_NO_MITSHM=1" $CLIENT_DOCKER
elif [ "$cpu" = true ]; then
    # Run docker command without GPU option
    docker run --network host -it --privileged \
    -v $FURNITURE_BENCH:/furniture-bench \
    --rm --ipc=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOST_DATA_MOUNT:$CONTAINER_DATA_MOUNT \
    --env="QT_X11_NO_MITSHM=1" $CLIENT_DOCKER
elif [ "$sim_gpu" = true ]; then
    # Verify if environment variable ISAAC_GYM_PATH is set
    if [ -z "$ISAAC_GYM_PATH" ]; then
        echo "ISAAC_GYM_PATH is not set"
        exit 1
    fi
    docker run --network host -it --privileged \
    -v $FURNITURE_BENCH:/furniture-bench \
    --gpus=all --rm --ipc=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOST_DATA_MOUNT:$CONTAINER_DATA_MOUNT \
    -v $ISAAC_GYM_PATH:/isaacgym \
    --env="QT_X11_NO_MITSHM=1" $CLIENT_DOCKER
else
    # No option provided
    echo "Provide an option: --gpu, --cpu, or --sim-gpu"
    exit 1
fi
