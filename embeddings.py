#!/usr/bin/env python3
"""
Embeddings visualization for reasoning traces.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Embed and visualize reasoning traces")
    parser.add_argument("--input", required=True, help="Path to CSV file with reasoning traces")
    parser.add_argument("--embedding-model", required=True, help="HuggingFace embedding model name")
    parser.add_argument("--output", required=True, help="Output file path")
    
    args = parser.parse_args()
    
    # Load embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model, device=device)
    print(f"Model loaded on device: {device}")
    
    # Load CSV
    df = pd.read_csv(args.input)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Generate embeddings for thinking_content
    print("Generating embeddings...")
    thinking_texts = df['thinking_content'].tolist()
    embeddings = []
    
    for i, text in enumerate(thinking_texts):
        if i % 100 == 0:
            print(f"Processing {i}/{len(thinking_texts)}")
        embedding = model.encode([text])
        embeddings.append(embedding[0])
    
    embeddings = np.array(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Reduce dimensionality with t-SNE
    print("Reducing dimensions with t-SNE...")
    reducer = TSNE(n_components=2, random_state=42, perplexity=10)
    embedding_2d = reducer.fit_transform(embeddings)
    print(f"t-SNE embeddings shape: {embedding_2d.shape}")
    
    # Create visualization color-coded by method
    methods = df['method'].unique()
    colors = plt.cm.Set1(range(len(methods)))
    
    plt.figure(figsize=(12, 8))
    for i, method in enumerate(methods):
        mask = df['method'] == method
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=[colors[i]], label=method, alpha=0.7)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Reasoning Trace Embeddings by Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and show plot
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()