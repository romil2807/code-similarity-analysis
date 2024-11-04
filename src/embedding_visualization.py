from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(embeddings: np.ndarray, labels: list):
    """
    Create TSNE visualization of code embeddings
    """
    # Reduce dimensions to 2D using TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
    plt.title('Code Embeddings Visualization using t-SNE')
    plt.colorbar(scatter)
    plt.savefig('embeddings_visualization.png')
    plt.close()
