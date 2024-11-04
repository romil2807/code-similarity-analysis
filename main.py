from src.clone_detection import load_and_sample_pairs
from src.embedding_visualization import process_and_visualize

def main():
    print("Step 1: Loading and sampling code pairs...")
    load_and_sample_pairs()
    
    print("\nStep 2: Processing embeddings and creating visualization...")
    process_and_visualize()
    
    print("\nDone! Check code_embeddings_tsne.png for visualization")

if __name__ == "__main__":
    main()
