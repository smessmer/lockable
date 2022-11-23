#!/bin/bash
#
# This script generates the README.md file based on the rustdoc documentation of the crate.
#
# Note that this currently only supports the following types of links:
# - [Name](crate::...)
#
set -e

# Use "cargo rdme" to generate the README.md file
cargo rdme --force

echo Success
