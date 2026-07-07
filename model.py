"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    """
    Build a simple MLP classifier with one hidden layer.
    
    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden layer
        num_classes: Number of output classes
    
    Returns:
        nn.Module: Neural network that maps (N, input_size) to (N, num_classes) logits
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )

# Step 2 - build_synthetic_dataset
def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    """
    Build a synthetic classification dataset with reproducible randomness.
    
    Args:
        num_samples: Number of samples to generate
        input_size: Dimension of each feature vector
        num_classes: Number of classes for labels
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (features, labels) where features is (num_samples, input_size) float tensor
               and labels is (num_samples,) long tensor with values in [0, num_classes)
    """
    # Create a seeded generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Generate random features from a standard normal distribution
    features = torch.randn(num_samples, input_size, generator=generator, dtype=torch.float)
    
    # Generate random labels uniformly from [0, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator, dtype=torch.long)
    
    return features, labels

# Step 3 - train_test_split_dataset
def train_test_split_dataset(features, labels, test_fraction, seed):
    """
    Split dataset into train and test sets with reproducible randomness.
    
    Args:
        features: (N, input_size) tensor of features
        labels: (N,) tensor of labels
        test_fraction: Fraction of data to use for test set
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_features, train_labels, test_features, test_labels)
    """
    N = features.shape[0]
    
    # Create a seeded generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Generate shuffled indices
    indices = torch.randperm(N, generator=generator)
    
    # Calculate test size
    test_size = int(N * test_fraction)
    
    # Split indices
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Split the data
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
def partition_data_iid(train_features, train_labels, num_clients, seed):
    """
    Partition training data across clients in an IID manner.
    
    Args:
        train_features: (M, input_size) tensor of training features
        train_labels: (M,) tensor of training labels
        num_clients: Number of clients to partition data among
        seed: Random seed for reproducibility
    
    Returns:
        list: List of (client_features, client_labels) tensor pairs for each client
    """
    M = train_features.shape[0]
    
    # If no clients specified, treat as 1 client (all data)
    if num_clients <= 0:
        num_clients = 1
    
    # Create a seeded generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Shuffle the row indices
    indices = torch.randperm(M, generator=generator)
    
    # Shuffle the data using the permuted indices
    shuffled_features = train_features[indices]
    shuffled_labels = train_labels[indices]
    
    # Calculate base samples per client and remainder
    samples_per_client = M // num_clients
    remainder = M % num_clients
    
    # Split data into contiguous groups for each client
    parts = []
    start_idx = 0
    
    for i in range(num_clients):
        # Distribute the remainder one by one to the first few clients
        extra = 1 if i < remainder else 0
        end_idx = start_idx + samples_per_client + extra
        
        # Extract client data
        client_features = shuffled_features[start_idx:end_idx]
        client_labels = shuffled_labels[start_idx:end_idx]
        
        parts.append((client_features, client_labels))
        
        start_idx = end_idx
    
    return parts

# Step 5 - partition_data_non_iid
def partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed):
    """
    Partition training data across clients in a non-IID manner.
    
    Each client receives shards_per_client label-contiguous shards, making
    their label distribution skewed (heterogeneous).
    
    Args:
        train_features: (M, input_size) tensor of training features
        train_labels: (M,) tensor of training labels
        num_clients: Number of clients to partition data among
        shards_per_client: Number of label shards each client receives
        seed: Random seed for reproducibility
    
    Returns:
        list: List of (client_features, client_labels) tensor pairs for each client
    """
    M = train_features.shape[0]
    
    # Handle edge cases
    if num_clients <= 0:
        num_clients = 1
    
    # Create a seeded generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Sort by labels
    sorted_indices = torch.argsort(train_labels)
    sorted_features = train_features[sorted_indices]
    sorted_labels = train_labels[sorted_indices]
    
    # Total shards needed
    total_shards = num_clients * shards_per_client
    
    # Split into shards
    shards = []
    samples_per_shard = M // total_shards
    
    for i in range(total_shards):
        start_idx = i * samples_per_shard
        end_idx = (i + 1) * samples_per_shard if i < total_shards - 1 else M
        
        shard_features = sorted_features[start_idx:end_idx]
        shard_labels = sorted_labels[start_idx:end_idx]
        shards.append((shard_features, shard_labels))
    
    # Shuffle shards
    shard_indices = torch.randperm(total_shards, generator=generator)
    
    # Assign shards to clients
    parts = []
    for client_idx in range(num_clients):
        client_features_list = []
        client_labels_list = []
        
        for s in range(shards_per_client):
            shard_idx = client_idx * shards_per_client + s
            if shard_idx < total_shards:
                actual_shard_idx = shard_indices[shard_idx]
                feat, lab = shards[actual_shard_idx]
                client_features_list.append(feat)
                client_labels_list.append(lab)
        
        # Concatenate
        if client_features_list:
            client_features = torch.cat(client_features_list, dim=0)
            client_labels = torch.cat(client_labels_list, dim=0)
            parts.append((client_features, client_labels))
        else:
            parts.append((torch.tensor([]), torch.tensor([], dtype=torch.long)))
    
    return parts

# Step 6 - count_client_samples
def count_client_samples(client_partitions):
    """
    Count the number of samples held by each client.
    
    Args:
        client_partitions: List of (client_features, client_labels) tensor pairs
    
    Returns:
        list: List of ints giving the number of samples for each client, in the same order
    """
    return [features.shape[0] for features, _ in client_partitions]

# Step 7 - iterate_client_batches
def iterate_client_batches(client_features, client_labels, batch_size, seed):
    """
    Shuffle one client's data and split it into mini-batches.
    
    Args:
        client_features: (n, input_size) tensor of client features
        client_labels: (n,) tensor of client labels
        batch_size: Size of each mini-batch
        seed: Random seed for reproducibility
    
    Returns:
        list: List of (batch_features, batch_labels) tuples covering all data exactly once
    """
    n = client_features.shape[0]
    
    # Handle empty client
    if n == 0:
        return []
    
    # Create a seeded generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Shuffle the row indices
    indices = torch.randperm(n, generator=generator)
    
    # Shuffle the data using the permuted indices
    shuffled_features = client_features[indices]
    shuffled_labels = client_labels[indices]
    
    # Split into batches
    batches = []
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_features = shuffled_features[start_idx:end_idx]
        batch_labels = shuffled_labels[start_idx:end_idx]
        batches.append((batch_features, batch_labels))
    
    return batches

# Step 8 - compute_batch_loss
def compute_batch_loss(model, batch_features, batch_labels):
    """
    Compute cross-entropy loss for one batch given the model.
    
    Args:
        model: nn.Module that maps features to logits
        batch_features: (B, input_size) tensor of features
        batch_labels: (B,) tensor of integer class labels
    
    Returns:
        torch.Tensor: Scalar loss tensor attached to computation graph
    """
    # Get logits from the model
    logits = model(batch_features)
    
    # Compute cross-entropy loss
    # The model returns raw logits, so we use CrossEntropyLoss which applies softmax internally
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, batch_labels)
    
    return loss

# Step 9 - local_sgd_step (not yet solved)
# TODO: implement

# Step 10 - train_client_local (not yet solved)
# TODO: implement

# Step 11 - clone_model_state (not yet solved)
# TODO: implement

# Step 12 - load_model_state (not yet solved)
# TODO: implement

# Step 13 - initialize_global_state (not yet solved)
# TODO: implement

# Step 14 - add_state_dicts (not yet solved)
# TODO: implement

# Step 15 - scale_state_dict (not yet solved)
# TODO: implement

# Step 16 - aggregate_weighted_average (not yet solved)
# TODO: implement

# Step 17 - select_round_clients (not yet solved)
# TODO: implement

# Step 18 - run_communication_round (not yet solved)
# TODO: implement

# Step 19 - evaluate_accuracy (not yet solved)
# TODO: implement

# Step 20 - run_fedavg (not yet solved)
# TODO: implement

# Step 21 - train_centralized_baseline (not yet solved)
# TODO: implement

# Step 22 - run_fedavg_iid (not yet solved)
# TODO: implement

# Step 23 - run_fedavg_non_iid (not yet solved)
# TODO: implement

# Step 24 - compute_non_iid_gap (not yet solved)
# TODO: implement

# Step 25 - rounds_to_target_vs_local_epochs (not yet solved)
# TODO: implement

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

