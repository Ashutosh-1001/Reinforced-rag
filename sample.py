import json
import random
from pathlib import Path

CORPUS = """
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
Unsupervised learning finds patterns in unlabeled data without explicit guidance.
Reinforcement learning trains agents to take actions in an environment to maximize cumulative reward.
Neural networks are computational models inspired by the structure of biological brains.
Deep learning uses neural networks with many layers to learn hierarchical representations.
Transformers are a neural network architecture based on self-attention mechanisms.
Self-attention allows each token in a sequence to attend to every other token.
BERT is a transformer model pretrained on masked language modeling and next sentence prediction.
GPT models are autoregressive transformers trained to predict the next token in a sequence.
Retrieval-Augmented Generation combines a retriever and a generator to answer questions.
The retriever fetches relevant documents from a knowledge base using embedding similarity.
The generator uses retrieved context to produce a grounded natural language answer.
FAISS is a library for efficient similarity search over dense vectors.
MMR stands for Maximal Marginal Relevance and balances relevance with diversity in retrieval.
Policy gradient methods optimize a policy by following the gradient of expected reward.
REINFORCE is a Monte Carlo policy gradient algorithm that uses full episode returns.
The reward function defines what behavior the agent should learn to maximize.
Cosine similarity measures the angle between two vectors and is bounded between -1 and 1.
Embeddings are dense vector representations of text learned by neural models.
"""

QA_PAIRS = [
    {"query": "What is supervised learning?", "answer": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs."},
    {"query": "How does FAISS work?", "answer": "FAISS is a library for efficient similarity search over dense vectors."},
    {"query": "What is RAG?", "answer": "Retrieval-Augmented Generation combines a retriever and a generator to answer questions using relevant documents from a knowledge base."},
    {"query": "What is REINFORCE?", "answer": "REINFORCE is a Monte Carlo policy gradient algorithm that uses full episode returns to optimize a policy."},
    {"query": "What is MMR?", "answer": "MMR stands for Maximal Marginal Relevance and balances relevance with diversity in retrieval."},
    {"query": "What are embeddings?", "answer": "Embeddings are dense vector representations of text learned by neural models."},
    {"query": "How does self-attention work?", "answer": "Self-attention allows each token in a sequence to attend to every other token."},
    {"query": "What is cosine similarity?", "answer": "Cosine similarity measures the angle between two vectors and is bounded between -1 and 1."},
    {"query": "What is deep learning?", "answer": "Deep learning uses neural networks with many layers to learn hierarchical representations."},
    {"query": "What is reinforcement learning?", "answer": "Reinforcement learning trains agents to take actions in an environment to maximize cumulative reward."},
]


def generate(output_dir="data"):
    Path(output_dir).mkdir(exist_ok=True)

    corpus_path = f"{output_dir}/sample_corpus.txt"
    with open(corpus_path, "w") as f:
        f.write(CORPUS.strip())
    print(f"Corpus written to {corpus_path}")

    random.shuffle(QA_PAIRS)
    split = int(len(QA_PAIRS) * 0.8)
    train_pairs = QA_PAIRS[:split]
    eval_pairs = QA_PAIRS[split:]

    train_path = f"{output_dir}/train_pairs.jsonl"
    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Train pairs written to {train_path} ({len(train_pairs)} examples)")

    eval_path = f"{output_dir}/eval_pairs.jsonl"
    with open(eval_path, "w") as f:
        for p in eval_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Eval pairs written to {eval_path} ({len(eval_pairs)} examples)")


if __name__ == "__main__":
    generate()
