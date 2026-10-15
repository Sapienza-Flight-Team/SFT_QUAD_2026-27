#!/usr/bin/env bash
set -e

echo "Checking pdm availability"
type -p pdm >/dev/null || curl -sSL https://pdm-project.org/install-pdm.py | python3 -

grep -qxF 'export PATH=$HOME/.local/bin:$PATH' "$HOME/.bashrc" || echo 'export PATH=$HOME/.local/bin:$PATH' >> "$HOME/.bashrc"
grep -qxF 'export PATH=$HOME/.local/bin:$PATH' "$HOME/.zshrc" || echo 'export PATH=$HOME/.local/bin:$PATH' >> "$HOME/.zshrc"

echo "Installing dependencies"
ARCH=$(uname -m)

if [ "$ARCH" = "aarch64" ]; then
    echo "Detected Jetson environment"
    pdm install -G jetson
else
    echo "Detected Laptop environment"
    read -r -p "Are you using an NVIDIA Pascal GPU or older? [y/N] " OLD_NVIDIA_GPU
    case "$OLD_NVIDIA_GPU" in
        [yY]|[yY][eE][sS])
            pdm install -G dev-cu126
            ;;
        *)
            pdm install -G dev
            ;;
    esac
fi

git submodule update --init --recursive
pdm run pre-commit install

echo "Everything done. Please restart your terminal"
