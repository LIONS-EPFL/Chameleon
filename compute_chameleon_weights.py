# compute_chameleon_weights.py

import os
import argparse
import torch
import numpy as np

# Import functions from the new modules
from slimpajama import get_slimpajama_6b
from utils import (
    initialize_model_and_tokenizer,
    get_random_samples,
    get_gpt2_embeddings_batched,
    compute_domain_affinity_matrix,
    compute_ridge_leverage_scores,
    load_and_prepare_embeddings,
    calculate_domain_average_embeddings
)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="CHAMELEON: Compute domain weights for LLM training/finetuning.")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["pretrain", "finetune"], 
        required=True,
        help="Mode of operation: 'pretrain' for general knowledge, 'finetune' for task-specific knowledge."
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="./checkpoints/BASE-82M/",
        help="Path to the proxy model checkpoint (e.g., an 82M parameter GPT-2 model)."
    )
    parser.add_argument(
        "--embeddings_dir", 
        type=str, 
        default="./embeddings/",
        help="Directory to save/load computed domain embeddings. Embeddings will be saved as <domain_name>_embeddings.pt."
    )
    parser.add_argument(
        "--lambda_reg", 
        type=float, 
        default=10.0,
        help="Regularization parameter for Kernel Ridge Leverage Scores."
    )
    parser.add_argument(
        "--ft_temperature", 
        type=float, 
        default=0.2,
        help="Temperature factor for softmax normalization of finetuning weights."
    )
    parser.add_argument(
        "--pt_temperature", 
        type=float, 
        default=5.0,
        help="Temperature factor for softmax normalization of pretraining weights."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=2048,
        help="Number of random samples to draw from each domain for embedding computation."
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64, # A reasonable default, adjust based on GPU memory
        help="Batch size for computing embeddings to manage GPU memory."
    )
    parser.add_argument(
        "--sample_length", 
        type=int, 
        default=512,
        help="Length (in tokens) of each sample used for embedding computation."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to run computations on (e.g., 'cuda', 'cpu', 'auto'). 'auto' will use cuda if available."
    )
    parser.add_argument(
        "--skip_embedding_generation", 
        action="store_true",
        help="If set, the script will skip generating new embeddings and try to load existing ones from --embeddings_dir. Useful for re-running weight calculation without re-computing embeddings."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Running computations on: {device}")

    # Initialize model and tokenizer (from utils.py)
    initialize_model_and_tokenizer(args.checkpoint_path, args.sample_length, device)

    # Define the order of domains as used in the paper.
    # This ensures consistent ordering of results and loaded files.
    DOMAIN_ORDER = ['arxiv', 'book', 'c4', 'cc', 'github', 'stackexchange', 'wikipedia']

    # --- Step 1: Generate and Save Domain Embeddings (if not skipped) ---
    if not args.skip_embedding_generation:
        os.makedirs(args.embeddings_dir, exist_ok=True)
        print(f"\n--- Generating and Saving Domain Embeddings to {args.embeddings_dir} ---")
        for domain_name in DOMAIN_ORDER:
            print(f"Processing domain: {domain_name}")
            
            # Load domain data (using the placeholder from data_loader.py)
            try:
                # Assuming get_slimpajama returns a dict like {'train': data_iterator}
                domain_data_iterator = get_slimpajama_6b(domain_name)['train'] 
            except Exception as e:
                print(f"Error loading data for {domain_name}: {e}. Skipping embedding generation for this domain.")
                continue

            # Get random text samples
            random_texts = get_random_samples(domain_data_iterator, args.num_samples, args.sample_length)
            
            if not random_texts:
                print(f"No valid samples obtained for {domain_name}. Skipping embedding computation.")
                continue

            # Compute embeddings
            # embeddings = get_gpt2_embeddings(random_texts, device)
            embeddings = get_gpt2_embeddings_batched(random_texts, device, args.embedding_batch_size)
            
            if embeddings.numel() == 0:
                print(f"No embeddings computed for {domain_name}. Skipping save.")
                continue

            # Save embeddings to disk
            torch.save(embeddings.cpu(), os.path.join(args.embeddings_dir, f"{domain_name}_embeddings.pt"))
            print(f"Saved {embeddings.shape[0]} embeddings for {domain_name}.")
    else:
        print(f"\n--- Skipping embedding generation. Loading existing embeddings from {args.embeddings_dir} ---")

    # --- Step 2: Load All Embeddings and Compute Domain Average Embeddings ---
    try:
        all_raw_embeddings, domain_labels_raw, initial_domain_name_map = \
            load_and_prepare_embeddings(args.embeddings_dir, args.num_samples)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure embeddings are generated or the path is correct. Exiting.")
        return
    except ValueError as e:
        print(e)
        print("Exiting.")
        return

    print("\n--- Calculating Domain Average Embeddings ---")
    # This function will also return an updated domain_name_map
    # which only includes domains that had embeddings successfully loaded.
    domain_average_embeddings, final_domain_name_map = calculate_domain_average_embeddings(
        all_raw_embeddings, domain_labels_raw, initial_domain_name_map
    )
    
    # Re-order domain_name_map to ensure consistent output order
    sorted_domain_names = [final_domain_name_map[i] for i in sorted(final_domain_name_map.keys())]

    # --- Step 3: Compute Domain Affinity Matrix ---
    print("\n--- Computing Domain Affinity Matrix ---")
    domain_affinity_matrix = compute_domain_affinity_matrix(domain_average_embeddings)
    print(f"Domain Affinity Matrix shape: {domain_affinity_matrix.shape}")

    # --- Step 4: Compute Kernel Ridge Leverage Scores (KRLS) ---
    print(f"\n--- Computing Kernel Ridge Leverage Scores with lambda_reg={args.lambda_reg} ---")
    leverage_scores = compute_ridge_leverage_scores(domain_affinity_matrix, args.lambda_reg)
    print(f"Raw Leverage Scores (S_lambda(D_i)): {leverage_scores}")

    # --- Step 5: Calculate and Print Domain Weights based on Mode ---
    if args.mode == "finetune":
        # Finetuning emphasizes domain-specific uniqueness, so use KRLS directly.
        ft_logits = leverage_scores / args.ft_temperature
        ft_weights = np.exp(ft_logits) / np.sum(np.exp(ft_logits))
        print(f"\n--- Finetuning Weights (Higher for unique domains, using temperature={args.ft_temperature}) ---")
        for i, domain_name in enumerate(sorted_domain_names):
            print(f"Domain '{domain_name}': {ft_weights[i]:.4f}")
        print("\nThese weights should be used during the finetuning stage to emphasize task-specific knowledge.")

    elif args.mode == "pretrain":
        # Pretraining emphasizes general knowledge, so use inverse KRLS.
        pt_logits = 1.0 / (leverage_scores + 1e-12) / args.pt_temperature
        pt_weights = np.exp(pt_logits) / np.sum(np.exp(pt_logits))
        print(f"\n--- Pretraining Weights (Higher for common domains, using temperature={args.pt_temperature}) ---")
        for i, domain_name in enumerate(sorted_domain_names):
            print(f"Domain '{domain_name}': {pt_weights[i]:.4f}")
        print("\nThese weights should be used during the pretraining stage to promote general knowledge learning.")
    
    print("\nCHAMELEON weight computation complete.")

if __name__ == "__main__":
    main()
