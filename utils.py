import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.preprocessing import KernelCenterer

# --- Global Variables for Model and Tokenizer (initialized externally) ---
tokenizer = None
model = None

def initialize_model_and_tokenizer(checkpoint_path, sample_length, device):
    """
    Initializes the tokenizer and the proxy language model.
    """
    global tokenizer, model
    print(f"Loading tokenizer and model from {checkpoint_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = sample_length
    
    model = GPT2Model.from_pretrained(checkpoint_path).eval()
    model.to(device)
    print(f"Model loaded successfully and moved to {device}.")

def decode_tokens(token_ids):
    """
    Converts a list of token IDs back into a human-readable string.
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Call initialize_model_and_tokenizer first.")
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)


def get_random_samples(data, num_samples=100, sample_length=512):
    """Randomly select and decode num_samples from the data."""
    indices = np.random.randint(0, len(data) - sample_length, size=num_samples)
    return [decode_tokens(data[idx:idx + sample_length]) for idx in indices]


def _get_single_batch_gpt2_embeddings(texts, device):
    """
    Internal helper to compute averaged embeddings for a single batch of texts
    using the middle hidden layer of the loaded GPT-2 model.
    """
    if not texts:
        return torch.tensor([])
    if tokenizer is None or model is None:
        raise RuntimeError("Model and/or Tokenizer not initialized. Call initialize_model_and_tokenizer first.")

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    middle_layer_idx = len(outputs.hidden_states) // 2
    return outputs.hidden_states[middle_layer_idx].mean(dim=1).cpu()

def get_gpt2_embeddings_batched(texts, device, batch_size):
    """
    Computes averaged embeddings for a list of texts in batches to manage GPU memory.

    Args:
        texts (List[str]): A list of text strings for which to compute embeddings.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to run the model on.
        batch_size (int): The number of texts to process in each batch.

    Returns:
        torch.Tensor: A tensor of shape (num_texts, embedding_dim) containing the
                      averaged embeddings for each text.
    """
    if not texts:
        return torch.tensor([])

    all_embeddings = []
    num_texts = len(texts)
    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"    Computing embeddings for batch {i//batch_size + 1}/{(num_texts + batch_size - 1)//batch_size}...")
        batch_embeddings = _get_single_batch_gpt2_embeddings(batch_texts, device)
        if batch_embeddings.numel() > 0:
            all_embeddings.append(batch_embeddings)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not all_embeddings:
        return torch.tensor([])

    return torch.cat(all_embeddings, dim=0)

def compute_domain_affinity_matrix(embeddings):
    """
    Computes the domain affinity matrix (Gram matrix) from domain embeddings.
    Uses a linear kernel (inner product).
    """
    affinity_matrix = embeddings @ embeddings.T
    centered_affinity_matrix = KernelCenterer().fit_transform(affinity_matrix)
    return centered_affinity_matrix

def compute_ridge_leverage_scores(K, lambda_reg=10):
    """
    Computes Kernel Ridge Leverage Scores (KRLS) from the domain affinity matrix.
    The formula is S_lambda(D_i) = [K(K + lambda_reg * k * I)^(-1)]_ii,
    where 'k' is the number of domains.
    
    Args:
        K (np.ndarray): The domain affinity matrix (Gram matrix) of shape (k, k),
                        where k is the number of domains.
        lambda_reg (float): Regularization parameter.
    
    Returns:
        np.ndarray: An array of shape (k,) containing the KRLS for each domain.
    """
    k = K.shape[0] # Number of domains
    
    # Compute the regularized inverse: (K + lambda_reg * k * I)^(-1)
    regularized_K = K + lambda_reg * k * np.eye(k)
    
    try:
        K_reg_inv = np.linalg.inv(regularized_K)
    except np.linalg.LinAlgError:
        print("Warning: Regularized matrix is singular. Adding small epsilon to diagonal for inverse.")
        regularized_K += np.eye(k) * 1e-9 # Add a tiny value to prevent singularity
        K_reg_inv = np.linalg.inv(regularized_K)
    
    # Compute the product K * (K + lambda_reg * k * I)^(-1)
    product = K @ K_reg_inv
    
    # Extract diagonal elements as leverage scores.
    leverage_scores = np.diag(product)
    
    return leverage_scores

def load_and_prepare_embeddings(folder_path, 
                                num_samples_per_domain):
    """
    Loads pre-computed domain embeddings from a folder and prepares them
    for domain embedding calculation.
    
    Args:
        folder_path (str): The directory containing the saved '.pt' embedding files.
        num_samples_per_domain (int): The number of embeddings to sample per domain.
    
    Returns:
        tuple[np.ndarray, np.ndarray, dict]: 
            - A concatenated NumPy array of all sampled embeddings.
            - A NumPy array of corresponding numerical domain labels.
            - A dictionary mapping numerical labels to domain names.
    """
    # Define the order of domains as used in the paper for consistency[cite: 134].
    DOMAIN_ORDER = ['arxiv', 'book', 'c4', 'cc', 'github', 'stackexchange', 'wikipedia']
    
    all_embeddings_data = []
    domain_labels = []
    domain_name_map = {} # Maps numerical label to domain name

    print(f"Loading embeddings from {folder_path}...")
    for i, domain_name in enumerate(DOMAIN_ORDER):
        file_path = os.path.join(folder_path, f"{domain_name}_embeddings.pt")
        if os.path.exists(file_path):
            domain_embeddings = torch.load(file_path).numpy()
            
            # Remove duplicate embeddings (e.g., if there were identical short texts)
            unique_embeddings = np.unique(domain_embeddings, axis=0)
            
            # Randomly select a subset of embeddings for the domain, up to the configured number.
            num_to_select = min(num_samples_per_domain, len(unique_embeddings))
            selected_indices = np.random.choice(len(unique_embeddings), size=num_to_select, replace=False)
            
            all_embeddings_data.append(unique_embeddings[selected_indices])
            domain_labels.extend([i] * num_to_select)
            domain_name_map[i] = domain_name
            print(f"  Loaded {num_to_select} embeddings for '{domain_name}'.")
        else:
            print(f"  Warning: Embeddings file not found for domain '{domain_name}' at {file_path}. Skipping.")
    
    if not all_embeddings_data:
        raise FileNotFoundError(f"No domain embedding files were loaded from {folder_path}. Please ensure embeddings are generated or the path is correct.")

    return np.concatenate(all_embeddings_data), np.array(domain_labels), domain_name_map

def calculate_domain_average_embeddings(full_embeddings_matrix, 
                                        domain_indices, 
                                        domain_name_map):
    """
    Computes the average embedding for each unique domain.
    """
    num_unique_domains = len(domain_name_map)
    domain_average_embeddings_list = []
    
    for i in range(num_unique_domains):
        domain_data_points = full_embeddings_matrix[domain_indices == i]
        if domain_data_points.size > 0:
            domain_average_embeddings_list.append(np.mean(domain_data_points, axis=0))
        else:
            domain_name = domain_name_map.get(i, f"Unknown Domain {i}")
            print(f"Warning: No data points found for domain '{domain_name}' (index {i}). This domain will not have an average embedding.")
            # Append a placeholder or handle as appropriate; here, we'll ensure the final array
            # matches the number of domains, even if some are None, and then filter.
            # A more robust approach might be to remove such domains from consideration entirely.
            domain_average_embeddings_list.append(None) 
    
    valid_averages = [avg for avg in domain_average_embeddings_list if avg is not None]
    if not valid_averages:
        raise ValueError("No valid average domain embeddings could be calculated for any domain.")
    
    # Reconstruct domain_name_map to only include domains for which averages were computed
    # This is important if some domains were skipped due to missing data.
    original_domain_names = [domain_name_map[k] for k in sorted(domain_name_map.keys())]
    final_domain_names = [name for i, name in enumerate(original_domain_names) if domain_average_embeddings_list[i] is not None]
    
    # Re-map numerical labels based on actual computed averages
    new_domain_name_map = {idx: name for idx, name in enumerate(final_domain_names)}

    return np.array(valid_averages), new_domain_name_map