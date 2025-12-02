## Neural Networks & Machine Learning Algorithms — From Scratch

This repository contains implementations of core machine learning and neural network models **built from scratch** using vanilla Python and NumPy.  
The goal is to provide a clear, low-level view of how these algorithms work internally, without relying on high-level frameworks such as TensorFlow, Keras, or PyTorch for the core logic.

### Main Topics

- **Multi-Layer Perceptron (MLP)**
  - Forward and backward propagation
  - Manual gradient computation
  - Custom activation functions and loss definitions
  - Training loops built step-by-step
  - 2-layer MLP with federated learning
  - 3-layer MLP with local training

- **Group Method of Data Handling (GMDH)**
  - 1-layer GMDH with RBF feature extraction
  - 1-layer GMDH with rough set theory (upper/lower bounds)

- **Autoencoders**
  - Encoder–decoder architectures
  - Manually coded backpropagation
  - Stacked autoencoders with local and global training
  - Autoencoder + MLP pipelines

- **Federated Learning**
  - Training across multiple clients
  - Global model aggregation
  - Privacy-preserving learning setups

### Project Layout

The repository is organized under a `src` directory:

- **`src/.ipynb`** – Jupyter notebooks for all experiments and models  
  - `gmdh_rbf_1layer.ipynb` – GMDH with RBF kernel (1 layer)  
  - `gmdh_rough_1layer.ipynb` – GMDH with rough set theory (1 layer)  
  - `mlp_2layer_federated_global.ipynb` – 2-layer MLP with federated global aggregation  
  - `mlp_3layer_local.ipynb` – 3-layer MLP with local training  
  - `data_normalization.ipynb` – data preprocessing and normalization utilities  
  - `autoencoder_mlp_multiple_models.ipynb` – multiple autoencoder + MLP combinations  
  - `mlp_autoencoder_multiple_models.ipynb` – MLP and autoencoder model variations  
  - Additional notebooks for GRU, LSTM, time-series forecasting, and autoencoder variants.

- **`src/python`** – Standalone Python scripts  
  - `ae3_mlp2_classification.py`  
  - `ae3_mlp2_regression.py`  
  - `Three layer AE by local train plus 3 layer MLP plus global train-Mohammad_Mohammadi.py`

- **`src/matlab`** – MATLAB implementations (`.m` files) of related homework and experiments.

- **`src/data`** – Input datasets used by the notebooks and scripts (`.csv`, `.xlsx`), for example:  
  - `Bias_correction_ucl.csv`, `Bias_correction_ucl(Normalized).csv`  
  - `Mackey-Glass.xlsx`, `melborn.xlsx`, `switzerlan universities.xlsx`

- **`src/docs`** – Reports and figures (`.pdf`, `.png`, `.docx`) such as homework write-ups and network diagrams.

### Notebook Documentation

All notebooks are written for learning and exploration:

- **Markdown cells** describe the motivation and theory behind each block of code  
- **Section headers** make it easy to follow the workflow (data prep → model → training → evaluation)  
- **Inline comments** explain key implementation details  
- **Plots and tables** visualize training progress and results

### Academic Context

These materials were developed as part of **Neural Networks (NN)** and **Deep Learning (DL)** coursework.  
They are intended to strengthen theoretical understanding by re-implementing algorithms in a transparent, step-by-step way.

### Technologies Used

- **Python 3.12+**  
- **NumPy** – numerical computations  
- **Matplotlib / Seaborn** – visualization  
- **Jupyter Notebook / JupyterLab** – interactive development  
- **TensorFlow/Keras** – used only for utilities such as loading datasets (e.g., MNIST)  
- **scikit-learn** – preprocessing helpers

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd Deep-Learning-Course-From-Scratch-codes-
```

2. **Create and activate a virtual environment (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

If you prefer using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or install the core libraries manually:

```bash
pip install numpy matplotlib tensorflow scikit-learn seaborn jupyter ipykernel
```

4. **Register a Jupyter kernel (optional)**

```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

5. **Launch Jupyter**

```bash
jupyter notebook   # or: jupyter lab
```

### How to Use the Project

- Open the notebooks from `src/.ipynb` in Jupyter  
- Select the correct kernel (for example, `Python (venv)`)  
- Run cells in order to follow the full data → model → training → evaluation pipeline  
- Notebooks are self-contained and can be run independently

### Example Notebook Highlights

- **`mlp_2layer_federated_global.ipynb`**  
  - Architecture: input (784) → hidden (32, ReLU) → output (10)  
  - Federated training across 3 clients, global aggregation, momentum-based optimization

- **`mlp_3layer_local.ipynb`**  
  - 3-layer MLP with manual backpropagation  
  - Custom activations and training loops

- **`gmdh_rbf_1layer.ipynb`**  
  - 1-layer GMDH + RBF network with polynomial feature extraction and neuron pruning  
  - Applied to Lorenz time-series data

- **`gmdh_rough_1layer.ipynb`**  
  - 1-layer GMDH + Rough Set Neural Network  
  - Interval-valued predictions with upper/lower bounds on Lorenz data

- **`autoencoder_mlp_multiple_models.ipynb`**  
  - Multiple configurations: 2-layer MLP, 3-layer MLP, and 3-layer Autoencoder + MLP (local and global training)

### Key Features

- **From-scratch implementations** – no high-level training abstractions  
- **Educational focus** – extensive use of markdown and comments  
- **Multiple architectures** – MLP, GMDH, autoencoders, GRU/LSTM, and federated learning setups  
- **Rich visualizations** – training curves, confusion matrices, regression plots, and more

### Contributing

This is primarily an academic/learning project, but you are welcome to:

- Open issues or questions  
- Suggest refactors or new experiments  
- Fork the repo and explore alternative architectures

### License

This project is shared for educational and research purposes.

### Acknowledgments

Developed as part of Neural Networks and Deep Learning coursework to better understand the fundamentals of modern machine learning algorithms.


