from datasets import load_dataset
import random
import json

def load_and_sample_pairs(num_pairs=5, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
    
    # Get samples from training set
    train_data = dataset['train']
    
    # Get some clone pairs (label=1) and non-clone pairs (label=0)
    clone_pairs = [item for item in train_data if item['label'] == 1]
    non_clone_pairs = [item for item in train_data if item['label'] == 0]
    
    # Randomly sample pairs
    selected_clones = random.sample(clone_pairs, num_pairs // 2)
    selected_non_clones = random.sample(non_clone_pairs, num_pairs - (num_pairs // 2))
    
    # Combine samples
    selected_pairs = selected_clones + selected_non_clones
    random.shuffle(selected_pairs)
    
    # Save samples to file
    with open('selected_samples.json', 'w') as f:
        json.dump(selected_pairs, f, indent=2)
    
    print(f"Selected {len(selected_pairs)} pairs and saved to selected_samples.json")
    return selected_pairs

if __name__ == "__main__":
    samples = load_and_sample_pairs()
    
    # Print sample information
    print("\nSample pairs summary:")
    for idx, pair in enumerate(samples, 1):
        print(f"\nPair {idx} (Clone: {'Yes' if pair['label'] == 1 else 'No'})")
        print(f"Code 1 length: {len(pair['code1'])}")
        print(f"Code 2 length: {len(pair['code2'])}")
