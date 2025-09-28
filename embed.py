import numpy as np
import os
from dotenv import load_dotenv
from gensim.models.keyedvectors import KeyedVectors

load_dotenv()
topic_embedding_model_path = os.getenv("TOPIC_EMBEDDING_MODEL_PATH")

word_vect = KeyedVectors.load_word2vec_format(topic_embedding_model_path, binary=True)

def get_topic_embedding(topic: str) -> np.array:
    """Get the embedding vector for a given topic using the pre-trained Word2Vec model."""
    # Handle non-string topics (NaN, float, None, etc.)
    if not isinstance(topic, str) or topic is None:
        # Return zero vector for invalid topics
        return np.zeros(word_vect.vector_size)

    # Handle empty strings
    if not topic.strip():
        return np.zeros(word_vect.vector_size)

    words = [w for w in topic.replace('-', ' ').replace('.', ' ').split() if w]
    embeddings = []
    for word in words:
        try:
            embeddings.append(word_vect.get_vector(word))
        except KeyError:
            continue
    if not embeddings:
        embeddings.append(np.zeros(word_vect.vector_size))
    return np.mean(embeddings, axis=0)


def get_all_topics_embedding(topic_to_idx: dict[str, int]) -> np.ndarray | None:
    """Generate embeddings for all topics."""
    embedding_dim = word_vect.vector_size
    embeddings = np.zeros((len(topic_to_idx) + 1, embedding_dim), dtype=np.float32)

    print("Generating topic embeddings...")

    for topic in topic_to_idx:
        idx = topic_to_idx[topic]
        embedding = get_topic_embedding(topic)
        embeddings[idx] = embedding
    print("Topic embeddings generated.")
    return np.array(embeddings, dtype=np.float32)


def convert_to_embedding(topics_values: np.ndarray, topic_to_idx: dict[str, int]) -> np.ndarray:
    """Convert topic indices to their corresponding embeddings."""
    embedding_dim = word_vect.vector_size
    embedded_vectors = np.zeros((topics_values.shape[0], embedding_dim), dtype=np.float32)

    topic_embeddings = get_all_topics_embedding(topic_to_idx)

    # Vectorized approach - much faster
    print(f"Converting {topics_values.shape[0]} entities to embeddings...")

    # Process in batches to avoid memory issues
    batch_size = 1000
    num_batches = (topics_values.shape[0] + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, topics_values.shape[0])

        batch_values = topics_values[start_idx:end_idx]

        # Vectorized computation for the batch
        for i, values in enumerate(batch_values):
            nonzero_indices = np.nonzero(values)[0]
            if len(nonzero_indices) > 0:
                # Get embeddings for non-zero topics and multiply by their values
                topic_embeds = topic_embeddings[nonzero_indices + 1]  # +1 for padding offset
                weights = values[nonzero_indices].reshape(-1, 1)
                embedded_vectors[start_idx + i] = np.sum(topic_embeds * weights, axis=0)

        # Print progress less frequently
        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Processed {min(end_idx, topics_values.shape[0])}/{topics_values.shape[0]} entities")

    print("Embedding conversion completed.")
    return embedded_vectors
