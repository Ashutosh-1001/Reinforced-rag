import torch
import torch.nn as nn
import torch.optim as optim
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        return self.layers(x)


class ReinforcedRAG:
    def __init__(self, data_path):
        self.load_text_data(data_path)
        self.setup_embedding_model()
        self.setup_vector_retriever()
        self.setup_policy_network()
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        self.minimum_docs = 2

    def load_text_data(self, filepath):
        loader = TextLoader(filepath)
        documents = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separator="\n",
            length_function=len,
        )
        self.document_chunks = splitter.split_documents(documents)
        print(f"Loaded {len(self.document_chunks)} document chunks from {filepath}")

    def setup_embedding_model(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def setup_vector_retriever(self):
        self.vector_store = FAISS.from_documents(
            self.document_chunks, self.embedding_model
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
        )

    def setup_policy_network(self):
        embedding_dim = 384
        state_dim = embedding_dim * 2
        self.policy_net = PolicyNetwork(state_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

    def get_state_vector(self, query, document):
        try:
            query_embedding = self.embedding_model.embed_query(query)
            doc_embedding = self.embedding_model.embed_query(document.page_content)
            combined_embedding = query_embedding + doc_embedding
            return torch.FloatTensor(combined_embedding)
        except Exception:
            return None

    def compute_reward(self, generated_answer, reference_answer):
        try:
            gen_emb = torch.FloatTensor(
                self.embedding_model.embed_query(generated_answer)
            )
            ref_emb = torch.FloatTensor(
                self.embedding_model.embed_query(reference_answer)
            )
            reward = torch.nn.functional.cosine_similarity(
                gen_emb.unsqueeze(0), ref_emb.unsqueeze(0)
            ).item()
            return max(0.0, reward)
        except Exception:
            return 0.0

    def rerank_documents(self, query, candidate_docs):
        states = [self.get_state_vector(query, doc) for doc in candidate_docs]
        valid_pairs = [(s, d) for s, d in zip(states, candidate_docs) if s is not None]

        if len(valid_pairs) < self.minimum_docs:
            return candidate_docs, None, None

        valid_states, valid_docs = zip(*valid_pairs)
        state_tensor = torch.stack(valid_states)

        scores = self.policy_net(state_tensor).squeeze(-1)
        ranked_indices = torch.argsort(scores, descending=True)
        ranked_docs = [valid_docs[i] for i in ranked_indices]

        probabilities = torch.softmax(scores, dim=0)
        return ranked_docs, probabilities, ranked_indices

    def train_on_query(self, query, reference_answer):
        try:
            candidate_docs = self.retriever.get_relevant_documents(query)
            if len(candidate_docs) < self.minimum_docs:
                return 0.0, 0.0

            ranked_docs, probabilities, ranked_indices = self.rerank_documents(
                query, candidate_docs
            )
            if probabilities is None:
                return 0.0, 0.0

            context_text = "\n".join([doc.page_content for doc in ranked_docs[:3]])
            generated_answer = self.llm([
                SystemMessage(content=f"Answer based on:\n{context_text}"),
                HumanMessage(content=query)
            ]).content

            reward = self.compute_reward(generated_answer, reference_answer)

            log_probs = torch.log(probabilities.gather(0, ranked_indices))
            loss = -torch.mean(log_probs) * reward

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            return loss.item(), reward

        except Exception as e:
            print(f"Training step error: {e}")
            return 0.0, 0.0

    def query(self, question):
        candidate_docs = self.retriever.get_relevant_documents(question)
        ranked_docs, _, _ = self.rerank_documents(question, candidate_docs)
        context_text = "\n".join([doc.page_content for doc in ranked_docs[:3]])
        return self.llm([
            SystemMessage(content=f"Answer based on:\n{context_text}"),
            HumanMessage(content=question)
        ]).content
