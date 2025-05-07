# Set default shell to fish
set shell := ["fish", "-c"]

# Default recipe to run when just is called without arguments
default:
    @just --list

# Sync and update dependencies using uv
sync:
    uv sync --all-extras -U --compile-bytecode

# Initialize LaminDB
init-lamindb:
     uv run lamin init --storage ./data/nbl-dataset/ --modules bionty
