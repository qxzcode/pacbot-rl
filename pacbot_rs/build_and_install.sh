#!/bin/bash
set -e

# This is from https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build the extension module (with optimizations) and install it into the current venv.
cd "$SCRIPT_DIR"
RUSTFLAGS='-C target-cpu=native' maturin develop --release

# Make sure we can import pacbot_rs.
echo "Testing installation..."
cd ..
RUST_BACKTRACE=1 python -c 'import pacbot_rs'

echo "All good!"
