import torch
import argparse
from src.rag import ReinforcedRAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    rag = ReinforcedRAG(args.data)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        rag.policy_net.load_state_dict(state_dict)
        rag.policy_net.eval()
        print(f"Loaded checkpoint: {args.checkpoint}\n")
    else:
        print("No checkpoint provided\n")

    print("ReinforcedRAG ready\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer = rag.query(query)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
