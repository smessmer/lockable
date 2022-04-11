#!/bin/bash
#
# This script generates the README.md file based on the rustdoc documentation of the crate.
#
# Note that this currently only supports the following types of links:
# - [StructName]
# - [StructName::method_name]
# other types (e.g. namespaces, different crates, ...) are not supported
#
set -e

# Use "cargo readme" to generate the README.md file
cargo readme > README.md

echo Success
