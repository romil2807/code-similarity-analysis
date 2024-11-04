# Code Similarity Analysis

This project analyzes code similarity using CodeBERT embeddings and t-SNE visualization. It processes pairs of code snippets, generates their embeddings using CodeBERT, and visualizes their similarities using t-SNE dimensionality reduction.

## Features

- Code embedding generation using Microsoft's CodeBERT
- Dimensionality reduction using t-SNE
- Visualization of code similarities
- Support for multiple programming languages
- Configurable sample size and visualization parameters

## Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- matplotlib
- numpy
- pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code-similarity-analysis.git
cd code-similarity-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
code-similarity-analysis/
│
├── requirements.txt
├── README.md
├── main.py
├── data/
│   └── selected_samples.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── embedding_generator.py
│   └── embedding_visualization.py
└── figures/
    └── tsne_visualization.png
```

## Usage

1. Prepare your code pairs dataset in JSON format:
```json
[
    {
        "func1": "def example1():\n    return True",
        "func2": "def example2():\n    return False",
        "label": false
    }
]
```

2. Run the analysis:
```bash
python main.py
```

3. Check the generated visualization in `figures/tsne_visualization.png`

## Configuration

You can modify the following parameters in `main.py`:
- `n_samples`: Number of code pairs to analyze
- `perplexity`: t-SNE perplexity parameter (automatically adjusted if too high)
- `random_state`: Seed for reproducibility

## Methodology

### 1. Data Loading
- Loads code pairs from the dataset
- Randomly samples a specified number of pairs
- Saves selected samples for consistency

### 2. Embedding Generation
- Uses CodeBERT to generate embeddings for each code snippet
- Processes code through tokenization and model inference
- Creates fixed-size vector representations

### 3. Visualization
- Applies t-SNE dimensionality reduction
- Creates 2D scatter plot of code similarities
- Color-codes points based on pair membership

## Results

The visualization shows:
- Clusters of similar code snippets
- Spatial relationships indicating code similarity
- Color-coded points distinguishing between first and second functions in pairs

## Limitations

- CodeBERT has a maximum sequence length of 512 tokens
- t-SNE perplexity must be less than the number of samples
- Processing large datasets may require significant computational resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft's CodeBERT team for the pre-trained model
- scikit-learn team for t-SNE implementation
- The open-source community for various dependencies

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/code-similarity-analysis
