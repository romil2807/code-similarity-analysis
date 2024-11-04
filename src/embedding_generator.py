from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

def get_code_embedding(code, tokenizer, model):
    # Tokenize code
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding as code representation
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embedding[0]

def process_and_visualize():
    # Load saved samples
    with open('selected_samples.json', 'r') as f:
        samples = json.load(f)
    
    # Debug: Print the structure of the first sample
    print("Sample structure:", json.dumps(samples[0], indent=2))
    
    # Load CodeBERT
    print("Loading CodeBERT...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    
    # Get embeddings for all code snippets
    embeddings = []
    labels = []
    pair_ids = []
    
    print("Getting embeddings...")
    for idx, pair in enumerate(samples):
        # Get embeddings for both snippets in the pair
        emb1 = get_code_embedding(pair['func1'], tokenizer, model)
        emb2 = get_code_embedding(pair['func2'], tokenizer, model)
        
        embeddings.extend([emb1, emb2])
        labels.extend([1 if pair['label'] else 0] * 2)
        pair_ids.extend([idx, idx])
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Apply t-SNE with adjusted perplexity
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot points
    colors = ['red' if label == 1 else 'blue' for label in labels]
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
    
    # Add annotations
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(f'Pair {pair_ids[i]+1}', (x, y))
    
    plt.title('t-SNE visualization of code embeddings')
    plt.legend(['Clone', 'Non-Clone'])
    plt.savefig('code_embeddings_tsne.png')
    plt.close()
    
    print("Visualization saved as code_embeddings_tsne.png")

if __name__ == "__main__":
    process_and_visualize()
