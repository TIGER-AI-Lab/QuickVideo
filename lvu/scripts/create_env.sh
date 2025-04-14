#!/bin/bash

if [ -d ".venv" ]; then
    yes | rm -r .venv
fi

uv sync
uv pip install flash-attn --no-build-isolation
