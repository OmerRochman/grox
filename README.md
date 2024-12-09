# Grox -  Score-based Diffusion Modelling in Jax

Grox is a Python package for score-based diffusion modelling in Jax. It is intended for use in research. Right now it is in the very early stages of development.

## Setting Up the Environment

1. **Create a Conda Environment**

   Open your terminal and run the following command to create a new conda environment (you can use `micromamba` instead of `conda`):

   ```bash
   conda create --name grox python=3.12
   ```

2. **Activate the Environment**

   Activate the newly created environment:

   ```bash
   conda activate grox
   ```

3. **Install the Module in Editable Mode**

   Navigate to the root directory of your module and run:

   ```bash
   pip install -e .
   ```

   This installs the module in editable mode, allowing you to make changes to the code and have them reflected without reinstalling.

## Additional Notes

- The `-e` flag in the `pip install` command stands for "editable," which is useful for development purposes.
