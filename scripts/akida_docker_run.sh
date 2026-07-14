#!/usr/bin/env bash
# Build (if needed) and run the Akida/MetaTF dev container with the repo
# bind-mounted, so edits/commits made from inside it land on the host repo.
# Plain-`docker` equivalent of docker-compose.akida.yml, for hosts without the
# compose plugin (e.g. a fresh Colima install: `brew install docker colima`,
# `colima start`, no `docker compose` subcommand by default).
#
# Usage:
#   scripts/akida_docker_run.sh                 # interactive shell
#   scripts/akida_docker_run.sh pytest -q        # run one command, then exit
#   scripts/akida_docker_run.sh python scripts/akida_verify.py --n-seeds 5
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=eia-akida:latest

docker build -f Dockerfile.akida -t "$IMAGE" .

docker run --rm -it \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" \
    "${@:-bash}"
