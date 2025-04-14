
if [[ "$0" == "${BASH_SOURCE[0]}" ]]; then
    echo This must be sourced.
    exit 1
fi

source ./.venv/bin/activate