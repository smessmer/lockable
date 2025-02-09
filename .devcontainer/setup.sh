#!/bin/bash

set -e
set -v

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

${SCRIPT_DIR}/setup_fish.sh
${SCRIPT_DIR}/install_rust_utils.sh