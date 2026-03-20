import json
import torch
import torch.nn.functional as F
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.rag import ReinforcedRAG, PolicyNetwork


def load_pairs(filepath):
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def cosine_reward(emb_model, answer, reference):
    gen_emb = torch.FloatTensor(emb_model.embed_query(answer))
    ref_emb = torch.FloatTensor(emb_model.embed_query(reference))
    score = F.cosine_similarity(gen_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
    return max(0.0, score)


class VanillaRAG:
    def __init__(self, data_path):
        loader = TextLoader(data_path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator="\n")
        chunks = splitter.split_documents(documents)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
        )
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    def query(self, question):
        docs = self.retriever.get_relevant_documents(question)
        context = "\n".join([d.page_content for d in docs[:3]])
        return self.llm([
            SystemMessage(content=f"Answer based on:\n{context}"),
            HumanMessage(content=question)
        ]).content


def evaluate(data_path, pairs_path, checkpoint_path=None):
    pairs = load_pairs(pairs_path)
    print(f"Evaluating on {len(pairs)} pairs...\n")

    baseline = VanillaRAG(data_path)
    baseline_rewards = []
    for p in pairs:
        answer = baseline.query(p["query"])
        r = cosine_reward(baseline.embedding_model, answer, p["answer"])
        baseline_rewards.append(r)
    avg_baseline = sum(baseline_rewards) / len(baseline_rewards)
    print(f"Baseline avg reward: {avg_baseline:.4f}\n")

    rag = ReinforcedRAG(data_path)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        rag.policy_net.load_state_dict(state_dict)
        rag.policy_net.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")

    trained_rewards = []
    for p in pairs:
        answer = rag.query(p["query"])
        r = cosine_reward(rag.embedding_model, answer, p["answer"])
        trained_rewards.append(r)
    avg_trained = sum(trained_rewards) / len(trained_rewards)
    print(f"Trained avg reward:  {avg_trained:.4f}\n")

    improvement = ((avg_trained - avg_baseline) / max(avg_baseline, 1e-8)) * 100
    print("=" * 40)
    print(f"Baseline:  {avg_baseline:.4f}")
    print(f"Trained:   {avg_trained:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    print("=" * 40)

    return {
        "baseline": avg_baseline,
        "trained": avg_trained,
        "improvement_pct": improvement,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    evaluate(args.data, args.pairs, args.checkpoint)
