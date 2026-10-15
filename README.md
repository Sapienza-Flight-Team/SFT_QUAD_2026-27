[![Master Branch CI/CD](https://github.com/Sapienza-Flight-Team/sft_cv_2024_2025/actions/workflows/master_cicd.yml/badge.svg)](https://github.com/Sapienza-Flight-Team/sft_cv_2024_2025/actions/workflows/master_cicd.yml)

# Sapienza Flight Team - Computer Vision

Welcome to the official repository of the **Computer Vision** subteam of the Sapienza Flight Team!

## Installation

This repository uses the Python package manager **`pdm`** to manage dependencies and virtual environments.
To install it, follow the guide on the [official website](https://pdm.fming.dev/).

Once you have `pdm`, the rest of the installation is very simple — just follow the steps below.

<details>
  <summary>Linux/MacOS (Recommended)</summary>
  <ul>
    <li>Clone the repository: <code>git clone https://github.com/Sapienza-Flight-Team/sft_cv_2024_2025</code></li>
    <li>Run the script <code>install_repo.sh</code></li>
  </ul>
</details>

<details>
  <summary>Manual Installation</summary>
  <ul>
    <li>Clone the repository: <code>git clone https://github.com/Sapienza-Flight-Team/sft_cv_2024_2025</code></li>
    <li>Open a terminal (Windows Terminal, PowerShell, shell, zsh, iTerm2, etc.)</li>
    <li>Run <code>pdm install</code> to install dependencies</li>
    <li>(Optional) Run <code>pdm run pre-commit install</code> to install automatic hooks that ensure your code meets quality standards before being pushed to the remote repository.</li>
  </ul>
</details>

### PDM Dependency Management

To manage Python dependencies, we use **PDM**. There are a few important things to know when using it:

1.  The only reference for compatible dependency versions is written in **`pdm.lock`**, which **must not be edited manually**.
    Instead, `pdm` generates it from `pyproject.toml` and installs dependencies in the virtual environment it creates.
2.  Useful commands for initialization and dependency updates include:
    * `pdm sync`: install packages from the lock file
    * `pdm update`: update the lock file and then run `pdm sync`
    * `pdm install`: write dependencies specified in `pyproject.toml` to the lock file, then run `pdm update`

## Automatic Tests and Quality Control

Code quality in this repository is ensured through automated systems that run at different stages of the development workflow:

1.  **Branching Policy**: Work is only done on the `dev` branch. Changes must go through a **pull request (PR)** before being merged into the `main` branch. Each pull request must be approved by at least one other team member.
2.  **Continuous Integration (CI)**: When a pull request is submitted, the code is automatically tested. These tests verify that:
    * The code passes our defined test suite.
    * It complies with formatting, typing, and quality standards.
    If any test fails, the pull request is suspended until the issue is resolved.
3.  **Local Hooks**: Hooks are provided (and automatically activated by the `install_repo.sh` script). These automated checks run on every commit, detecting or correcting issues before they reach the remote repository.

### Pre-commit Hook

Pre-commit hooks check formatting, typing, and best practices on all modified files.

**Important**: If a file like `script.py` is automatically fixed, the change will **not** be added to the commit automatically. You must manually add it again using `git add script.py` (or `git add .` for all files).

Since hooks require manual attention, it is recommended **not to commit from graphical interfaces** (e.g., VS Code GUI). Instead, commit from the terminal to monitor the output of all checks.

## Documentation

All classes and methods are documented with **docstrings**.

Most IDEs (including VS Code) can display information about all symbols defined in `sft_cv` by hovering the cursor over them. General documentation is available in the **`docs`** folder.

## Running the Pipeline, Taking Pictures, and Saving Them

These steps are for running the pipeline on the **Jetson** (the onboard flight computer).

1.  **Turn on the Jetson**
    Plug in the AC adapter and wait for it to boot. The fan will spin at full speed after boot — this is normal.

2.  **Connect to the Jetson via SSH**
    Connect a Linux machine or a Windows machine with WSL to the Jetson using a USB cable (any port works).
    Open a terminal and run:
    ```bash
    ssh sft@jetorin.local
    ```
    Enter the password `sft` when prompted.

    > **Tip**: Once logged in via SSH, remember to run **`tmux`** so that when the drone flies away, the process isn’t killed.

3.  **Make sure the code on the Jetson is up to date**
    Navigate to the repository by running:
    ```bash
    cd sft_cv_2024_2025
    ```
    Then type:
    ```bash
    git branch
    ```
    to make sure you’re on the `master` branch. (All flight tests should be run from the master branch unless otherwise specified.)

    Then run:
    ```bash
    git pull
    ```
    If Git says the repository is up to date, you’re good to go.

### Tweak Parameters

Run:
```bash
~/tools/yaml_creator.py
```
You can change acquisition time, maximum exposure, and other parameters here.

### Running with USB Storage

When acquiring pictures with a USB stick, additional steps are required depending on your operating system before running the script:

<details>
  <summary>Linux</summary>
  <ul>
    <li>Mount the USB stick with <code>sudo mount /dev/sda1 /mnt/pdv_drive</code> (note: the device name could be <code>sdb1</code>, <code>sdc1</code>, etc.)</li>
  </ul>
</details>

Then, execute the pipeline to save to the USB:
```bash
screen bash on_usb_in_jetson.sh
```

The USB drive will contain all photos in its `outs` directory.
