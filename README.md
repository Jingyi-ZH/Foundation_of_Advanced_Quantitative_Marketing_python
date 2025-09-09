# Discrete Choice Models

This project is based on the "Foundations of Advanced Quantitative Marketing" course by [Professor Pradeep K. Chintagunta](https://www.chicagobooth.edu/faculty/directory/c/pradeep-k-chintagunta). It implements several discrete choice models commonly used in econometrics and demand estimation, including:
- Plain Logit Model
- Nested Logit Model
- Latent Class Model
- Random Coefficient Model
- BLP (Berry-Levinsohn-Pakes) Model

The project includes theoretical explanations, Python code implementations, and Jupyter notebooks for testing with simulated data and real-world datasets. It is designed for educational and research purposes, allowing users to understand, simulate, and apply these models.

## Folder Structure

- **src/**: Contains the core Python code for the models. Note that `logit.py` follows the logic as described in the lectures, which can better help beginners understand the model. While `logit_boost.py` provides an accelerated version of the model, making the implementation process significantly faster than the former.
- **Notebooks/**: Jupyter notebooks demonstrating the usage of the models, including:
  - Simulations for testing code.
  - Applications with real data.
- **Data/**: Datasets used in the notebooks.
- **Lectures/**: Theoretical sections, documents explaining the models' foundations, derivations, and assumptions.

## Requirements

- Python 3.8 or higher
- Required libraries (install via `pip install -r requirements.txt`, or manually):

## Usage

1. **Run the Notebooks**:
   - Navigate to the `Notebooks/` folder.
   - Start Jupyter Notebook or JupyterLab:
     ```
     jupyter notebook
     ```
   - Open the relevant `.ipynb` files for simulations or real data examples.
   - Each notebook includes step-by-step code execution, from data loading to model estimation.

2. **Using the Models in Code**:
   - Import models from `src/` in your Python scripts or notebooks, e.g.:
     ```python
     from src.logit import BLP
     blp_model = BLP(X, Z, shares, outside, D, t, p)
     gamma_hat, beta_hat = blp_model.fit()
     print(blp_model.summary())
     ```
   - Refer to the docstrings in the source code for detailed parameters and methods.

3. **Theory Reference**:
   - Check the `Lectures/` folder for detailed explanations of each model's theory, including background, mathematical formulations and assumptions.

## Examples

- **Simulation Notebook** (e.g., `simulation_rcm.ipynb` and `simulation_blp.ipynb` in `Notebooks/`):
  - Generates synthetic data.
  - Tests model estimation and compares results to true parameters.
  
- **Real Data Notebook** (e.g., `data_logit.ipynb` and `data_logit_boost.ipynb` in `Notebooks/`):
  - Loads data from `Data/` (e.g., consumer choice datasets).
  - Applies models like Nested Logit or BLP for demand estimation.

For specific model details:
- Plain Logit: Basic multinomial choice.
- Nested Logit: Accounts for correlated alternatives.
- Latent Class: Handles unobserved heterogeneity via classes.
- Random Coefficient: Incorporates individual-level variation.
- BLP: Instrumental variables approach for endogeneity in demand models.

## TODO

- Add an introduction to the data formats and parameters required by the models.
- Further optimization of the optimizer.

## Contact

If you have questions or suggestions, feel free to open an issue on the repository.