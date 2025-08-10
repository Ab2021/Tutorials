# Day 2.2: Comprehensive Environment Setup - A Practical Guide

## Introduction: A Solid Foundation is Key

A clean, reproducible, and powerful development environment is not a luxury; it's a necessity for any serious data science or deep learning project. Spending time setting up your environment correctly at the beginning will save you countless hours of debugging and dependency headaches later.

This guide provides a step-by-step walkthrough for setting up a professional Python environment for deep learning using **Conda**. We will cover creating isolated environments, installing PyTorch with GPU support, managing packages, and integrating with Jupyter Notebooks.

**Why Conda?**

*   **Environment Management:** Conda allows you to create isolated environments, so the packages for one project don't interfere with the packages for another.
*   **Package Management:** Conda can install Python packages (like `pip`) but also non-Python libraries and dependencies (like NVIDIA's CUDA toolkit), which is crucial for GPU support.
*   **Cross-Platform:** It works seamlessly on Windows, macOS, and Linux.

**Today's Learning Objectives:**

1.  **Install Miniconda:** A lightweight, bootstrap version of the Anaconda distribution.
2.  **Master Essential Conda Commands:** Learn to create, activate, list, and delete environments.
3.  **Install PyTorch with GPU Support:** Understand the relationship between your NVIDIA driver, the CUDA toolkit, and the PyTorch installation command.
4.  **Manage Packages:** Learn the best practices for using `conda` and `pip` together.
5.  **Set up Jupyter Notebooks:** Configure Jupyter to work within your isolated Conda environments.
6.  **Create an `environment.yml` file:** Learn how to export your environment so that others (or your future self) can replicate it perfectly.

---

## Part 1: Installing Miniconda

We recommend **Miniconda** over the full Anaconda distribution because it's much smaller and installs only the essentials. You can then install only the packages you need.

1.  **Go to the Miniconda documentation:** [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2.  **Download the installer** for your operating system (Windows, macOS, or Linux). Choose the latest Python 3.x version.
3.  **Run the installer.**
    *   **On Windows:** Double-click the `.exe` file. It is highly recommended to **accept the default settings**, especially the one that says "Add Miniconda3 to my PATH environment variable." While the installer warns against this, it makes using `conda` from standard terminals (like Command Prompt or PowerShell) much easier.
    *   **On macOS/Linux:** Open a terminal and run the script you downloaded, e.g., `bash Miniconda3-latest-Linux-x86_64.sh`. Follow the on-screen prompts. Say "yes" to running `conda init`.

4.  **Verify the installation:** Close and reopen your terminal (on Windows, this might be the "Anaconda Prompt" if you didn't add it to PATH). Type the following command:

    ```bash
    conda --version
    ```

    If it prints a version number (e.g., `conda 23.7.4`), your installation was successful.

---

## Part 2: Managing Conda Environments

This is the core strength of Conda. An environment is a self-contained directory that holds a specific collection of packages.

### 2.1. Creating an Environment

Let's create a new environment for our deep learning projects. We'll call it `pytorch_env` and install Python 3.10 in it.

```bash
# The -n flag specifies the name of the environment.
# We specify the python version we want to be installed.
conda create -n pytorch_env python=3.10
```

Conda will show you the packages it will install and ask you to proceed (`y/n`). Type `y` and press Enter.

### 2.2. Activating an Environment

To use an environment, you must "activate" it. This modifies your shell's PATH to point to the executables and libraries within that environment.

```bash
conda activate pytorch_env
```

After running this, you should see the name of the environment in your terminal prompt, e.g., `(pytorch_env) C:\Users\...>`.

### 2.3. Deactivating an Environment

When you are finished working, you can deactivate the environment to return to your base shell.

```bash
conda deactivate
```

### 2.4. Listing Environments

To see all the environments you have created:

```bash
conda env list
```

An asterisk (`*`) will indicate the currently active environment.

### 2.5. Deleting an Environment

If you no longer need an environment, you can remove it completely.

```bash
# Make sure you are not currently inside the environment you want to delete.
# First, run `conda deactivate` if needed.
conda env remove -n pytorch_env
```

---

## Part 3: Installing PyTorch with GPU Support (The Critical Step)

To use your NVIDIA GPU, PyTorch needs to be compiled against the correct version of NVIDIA's **CUDA Toolkit**. This is the most common point of failure for beginners.

**The Golden Rule:** Use the official PyTorch website's command generator!

1.  **Check your NVIDIA Driver Version:**
    *   Open a terminal and run `nvidia-smi`.
    *   Look at the top right of the output. It will show a "CUDA Version" (e.g., `CUDA Version: 12.1`). This is the **maximum** version of CUDA your driver supports. You can install any version of the CUDA toolkit *up to* this version.

2.  **Go to the PyTorch Website:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

3.  **Use the Interactive Tool:**
    *   **PyTorch Build:** Choose the Stable version.
    *   **Your OS:** Select your operating system.
    *   **Package:** Choose **Conda**.
    *   **Language:** Choose Python.
    *   **Compute Platform:** This is the most important part. Select the CUDA version you want to use (e.g., CUDA 11.8 or CUDA 12.1). It's generally safe to pick the latest one that is less than or equal to the version shown by `nvidia-smi`.

4.  **Run the Generated Command:** The website will generate a command. It will look something like this:

    ```bash
    # THIS IS AN EXAMPLE! DO NOT COPY-PASTE. GENERATE YOUR OWN!
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

5.  **Execute the command** inside your **activated** `pytorch_env` environment.

    ```bash
    # First, make sure you are in the right environment!
    conda activate pytorch_env

    # Now, run the command from the PyTorch website
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

    This command tells Conda to install PyTorch and its associated libraries (`torchvision`, `torchaudio`) along with the specific `pytorch-cuda` package from the correct channels (`-c pytorch -c nvidia`). Conda will handle downloading and setting up the CUDA toolkit for you within the environment. You do **not** need to install the CUDA toolkit system-wide.

### 3.1. Verifying the GPU Installation

After the installation is complete, verify that PyTorch can see your GPU.

```bash
# Make sure you are still in your activated environment
python
```

This will open a Python interpreter. Now, type the following:

```python
import torch

# Check if CUDA is available
is_available = torch.cuda.is_available()
print(f"Is CUDA available? {is_available}")

if is_available:
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
```

If `is_available` is `True`, your installation is successful! If it's `False`, something went wrong. The most common cause is an incompatibility between the NVIDIA driver version and the CUDA version you chose, or running the install command outside the correct Conda environment.

---

## Part 4: Managing Packages with Conda and Pip

**Best Practice:**
1.  Always try to install packages using `conda install <package_name>` first.
2.  If a package is not available in the Conda channels, use `pip install <package_name>`.

Conda and pip work well together, but it's best to install as much as possible with Conda to ensure binary compatibility, especially for scientific packages.

Let's install some common data science libraries:

```bash
# Activate the environment first
conda activate pytorch_env

# Install packages from conda-forge, a popular community channel
conda install -c conda-forge scikit-learn pandas matplotlib seaborn

# Install a package that might not be on conda, using pip
pip install some-other-library
```

---

## Part 5: Setting up Jupyter Notebooks

To use Jupyter Notebooks within your isolated environment, you need to install it *inside* that environment.

1.  **Install Jupyter:**

    ```bash
    conda activate pytorch_env
    conda install -c conda-forge notebook ipykernel
    ```
    `ipykernel` is the package that allows Jupyter to connect to the Python kernel of your specific environment.

2.  **Register the Environment with Jupyter:**

    ```bash
    python -m ipykernel install --user --name=pytorch_env --display-name="PyTorch Env (Python 3.10)"
    ```
    This command creates a "kernelspec" that tells Jupyter how to find and use the Python interpreter in your `pytorch_env`.

3.  **Launch Jupyter Notebook:**

    ```bash
    # You can launch it from your activated environment
    jupyter notebook
    ```

    When you create a new notebook (via the "New" button), you will now see "PyTorch Env (Python 3.10)" as an option in the dropdown menu. Choosing this ensures your notebook runs with the packages and Python version from your isolated environment.

---

## Part 6: Reproducibility - Exporting Your Environment

Once your environment is set up, you should create a file that lists all the packages and their exact versions. This allows anyone to replicate your environment perfectly.

```bash
# Activate the environment you want to export
conda activate pytorch_env

# Export the environment to a YAML file
conda env export > environment.yml
```

This creates a file named `environment.yml`. You can share this file with others. They can then create an identical environment with a single command:

```bash
# Someone else can now run this command to get the exact same environment
conda env create -f environment.yml
```

This is the standard and best practice for ensuring reproducibility in data science projects.

## Conclusion

You now have a professional, robust, and reproducible deep learning environment. You have learned how to isolate project dependencies, install the correct GPU-enabled version of PyTorch, and integrate with Jupyter Notebooks. This stable foundation will allow you to focus on what really matters: building and training amazing deep learning models.

## Self-Assessment Questions

1.  **Isolation:** Why is it a good idea to use a separate Conda environment for each major project?
2.  **`nvidia-smi`:** What is the most important piece of information you get from the `nvidia-smi` command when preparing to install PyTorch?
3.  **The Install Command:** Why should you always generate the PyTorch installation command from their official website?
4.  **Verification:** What is the one-line Python command to check if PyTorch can use your GPU?
5.  **Jupyter Kernel:** Why do you need to install `ipykernel` inside your Conda environment?
6.  **Reproducibility:** What is the name of the file used to share a Conda environment configuration, and what is the command to create an environment from that file?

