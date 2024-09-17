import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import support_files.OpenaiApi as OpenAIApi


def extract_text_with_cosine_similarityTEST(embeddings_ref, embeddings_sent, query, threshold, references, sentences):
    openAI = OpenAIApi.OpenaiApi()
    query_vector = openAI.vectorize_string(query)
    query_vector = np.array(query_vector)
    query_vector = query_vector.reshape(1, -1)

    # Compute cosine similarities
    similarity_to_query_ref = cosine_similarity(embeddings_ref, query_vector)
    similarity_to_query_sent = cosine_similarity(embeddings_sent, query_vector)

    # Find indices where similarity exceeds the threshold
    ref_indexes = np.where(similarity_to_query_ref > threshold)[0]
    sent_indexes = np.where(similarity_to_query_sent > threshold)[0]

    # Extract texts and vectors that meet the similarity threshold
    ref_similarity_text = [references[i] for i in ref_indexes]
    sent_similarity_text = [sentences[i] for i in sent_indexes]

    return sent_similarity_text, ref_similarity_text


def extract_text_with_cosine_similarity(embeddings, query, threshold, sentences):
    openAI = OpenAIApi.OpenaiApi()
    query_vector = openAI.vectorize_string(query)
    query_vector = np.array(query_vector)
    query_vector = query_vector.reshape(1, -1)

    # Compute cosine similarities
    similarity_to_query = cosine_similarity(embeddings, query_vector)

    # Find indices where similarity exceeds the threshold
    similarity_indexes = np.where(similarity_to_query > threshold)[0]

    # Extract texts and vectors that meet the similarity threshold
    similarity_text = [sentences[i] for i in similarity_indexes]

    return similarity_text
