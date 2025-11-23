# Neural Networks & Machine Learning Algorithms — From Scratch

This repository contains a collection of Jupyter Notebook (`.ipynb`) implementations of core machine learning and neural network models **built entirely from scratch** using vanilla Python and NumPy.  
The purpose of these implementations is to provide a deep, fundamental understanding of how these algorithms work internally, without relying on high-level libraries such as TensorFlow, Keras, or PyTorch.

## 📘 Contents

The repository includes manual implementations of:

### 🔷 Multi-Layer Perceptron (MLP)
- Forward and backward propagation
- Manual gradient computation
- Activation functions and custom loss definitions
- Training loops built step-by-step
- **2-layer MLP** with federated learning
- **3-layer MLP** with local training

### 🔷 Group Method of Data Handling (GMDH)
- **GMDH with RBF**: 1-layer GMDH feature extraction combined with Radial Basis Function networks
- **GMDH with Rough Set Theory**: 1-layer GMDH with rough set neural networks using upper and lower bounds

### 🔷 Autoencoders
- Encoder–decoder architecture implemented from scratch
- Backpropagation written manually
- Stacked autoencoders with local and global training
- Reconstruction error analysis and visualization
- Autoencoder + MLP combinations

### 🔷 Federated Learning
- Distributed training across multiple clients
- Global model aggregation
- Privacy-preserving machine learning

## 📂 Project Structure

### Root Directory
- `gmdh_rbf_1layer.ipynb` - GMDH with RBF kernel (1 layer)
- `gmdh_rough_1layer.ipynb` - GMDH with rough set theory (1 layer)
- `mlp_2layer_federated_global.ipynb` - 2-layer MLP with federated learning (global aggregation)

### DL Directory
- `mlp_3layer_local.ipynb` - 3-layer MLP with local training
- `data_normalization.ipynb` - Data preprocessing and normalization utilities
- `autoencoder_mlp_multiple_models.ipynb` - Multiple autoencoder and MLP model combinations
- `mlp_autoencoder_multiple_models.ipynb` - MLP and autoencoder model variations

## 📝 Notebook Documentation

All notebooks include:
- **Markdown cells** before each code cell explaining the purpose and functionality
- **Clear section headers** for easy navigation
- **Detailed comments** in code explaining implementation details
- **Visualizations** for training progress and results

## 🎓 Academic Context

These notebooks are practice material developed as part of **Neural Networks (NN)** and **Deep Learning (DL)** academic coursework.  
They aim to reinforce theoretical concepts by re-building the algorithms in a low-level, transparent manner.

## 🛠️ Technologies Used

- **Python 3.12+**
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development
- **TensorFlow/Keras** - For loading datasets (MNIST)
- **scikit-learn** - Data preprocessing utilities
- **Seaborn** - Statistical visualizations

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Deep-Learning-Course-From-Scratch-codes-
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install numpy matplotlib tensorflow scikit-learn seaborn jupyter ipykernel
```

### 4. Register Jupyter Kernel (Optional)
```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

## 📖 Usage

1. Open any `.ipynb` file in Jupyter Notebook or JupyterLab
2. Select the appropriate kernel (Python 3 or your virtual environment)
3. Run cells sequentially to understand the implementation
4. Each notebook is self-contained and can be run independently

## 🔍 Notebook Details

### Neural Network Models

#### `mlp_2layer_federated_global.ipynb`
- **Architecture**: Input (784) → Hidden (32, ReLU) → Output (10)
- **Training**: Federated learning with 3 clients
- **Dataset**: MNIST handwritten digits
- **Features**: Momentum-based optimization, global aggregation

#### `mlp_3layer_local.ipynb`
- **Architecture**: 3-layer MLP
- **Training**: Local training approach
- **Features**: Custom activation functions, manual backpropagation

#### `gmdh_rbf_1layer.ipynb`
- **Architecture**: 1-layer GMDH + RBF network
- **Features**: Polynomial feature extraction, neuron pruning
- **Dataset**: Lorenz time-series data

#### `gmdh_rough_1layer.ipynb`
- **Architecture**: 1-layer GMDH + Rough Set Neural Network
- **Features**: Interval-valued predictions, upper/lower bounds
- **Dataset**: Lorenz time-series data

### Autoencoder Models

#### `autoencoder_mlp_multiple_models.ipynb`
- Multiple model combinations:
  - 2-layer MLP
  - 3-layer MLP
  - 3-layer Autoencoder + 2-layer MLP (local/global)
  - 3-layer Autoencoder + 3-layer MLP (local/global)

#### `mlp_autoencoder_multiple_models.ipynb`
- Various MLP and autoencoder architectures
- Local and global training strategies

## 📊 Key Features

- ✅ **From Scratch Implementation**: No high-level ML frameworks
- ✅ **Educational Focus**: Detailed explanations and comments
- ✅ **Well Documented**: Markdown cells explain each step
- ✅ **Multiple Architectures**: MLP, GMDH, Autoencoders, Federated Learning
- ✅ **Visualizations**: Training curves, confusion matrices, regression plots

## 🤝 Contributing

This is an academic project. Feel free to:
- Report issues
- Suggest improvements
- Fork and experiment with different architectures

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

Developed as part of Neural Networks and Deep Learning coursework to understand the fundamentals of machine learning algorithms.
