#!/bin/bash

# Set up variables that are local to a given Orin/Bot
export CAMERA_FORWARD=39004209
export INFLUXDB_KEY="XXXXXX"
export INFLUXDB_BUCKET="paper_bucket"

docker-compose -f docker_compose_lqr.yaml --project-directory ./ up -d
