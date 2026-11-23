#!/bin/bash
#
# This script generates the README.md file based on the rustdoc documentation of the crate.
#
# Note that this currently only supports the following types of links:
# - [Name](crate::...)
#
set -e

cd "$(dirname "$0")"

# "cargo rdme" needs a nightly toolchain to resolve intralinks. We pin it in .cargo-rdme.toml
# (see there for why), make sure it is installed.
TOOLCHAIN=$(sed -n 's/^rustdoc-toolchain *= *"\(.*\)"$/\1/p' .cargo-rdme.toml)
rustup toolchain install "$TOOLCHAIN" --profile minimal --no-self-update

# Use "cargo rdme" to generate the README.md file
cargo rdme --force

echo Success
