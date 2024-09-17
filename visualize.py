from sklearn.manifold import TSNE
import numpy as np
import support_files.handle_data as data_handler
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import support_files.OpenaiApi as OpenAi
import support_files.bert as bert
import support_files.handle_vectors as vector_handler


def plot_diagram_with_query(doc_id, nr, threshold=0.75, query_text=None):
    diagram_title = f"Visualization on cosine similarity on document {nr}"

    Ai = OpenAi.OpenaiApi()
    query_vector = Ai.vectorize_string(text=query_text)

    embedding_sentences, embedding_references, text_sentences, text_references = data_handler.load_embeddings_and_text(doc_id)
    # Ensure embedding_ref is a numpy array
    embedding_ref = np.array(embedding_references)

    # Ensure embedding_sent is a numpy array
    embedding_sent = np.array(embedding_sentences)

    query_vector = np.array(query_vector)  # Ensure it's an array, if not already
    if query_vector.ndim == 1:  # Check if it's one-dimensional
        query_vector = query_vector.reshape(1, -1)  # Reshape to a single row with the same number of columns

    # Now, combine the flattened arrays for t-SNE
    combined_embeddings = np.vstack((embedding_ref, embedding_sent, query_vector))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    combined_tsne_embeddings = tsne.fit_transform(combined_embeddings)

    # Separate the transformed embeddings
    tsne_embeddings_ref = combined_tsne_embeddings[:len(embedding_ref)]
    tsne_embeddings_sent = combined_tsne_embeddings[len(embedding_ref):-1]
    query_emb = combined_tsne_embeddings[-1]

    # Assuming your existing setup is already correctly importing necessary libraries and loading data
    # Calculate cosine similarities
    similarity_to_query_ref = cosine_similarity(embedding_ref, query_vector)
    similarity_to_query_sent = cosine_similarity(embedding_sent, query_vector)

    # Find indices where the similarity exceeds the threshold
    similar_ref_indices = np.where(similarity_to_query_ref > threshold)[0]
    similar_sent_indices = np.where(similarity_to_query_sent > threshold)[0]

    # Add traces for each set of embeddings with hover text
    # Update marker color dynamically based on similarity threshold
    colors_ref = ['blue' if vector in similar_ref_indices else 'turquoise' for vector in range(len(tsne_embeddings_ref))]
    colors_sent = ['red' if vector in similar_sent_indices else 'pink' for vector in range(len(tsne_embeddings_sent))]

    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tsne_embeddings_ref[:, 0], y=tsne_embeddings_ref[:, 1], mode='markers',
                             marker=dict(color=colors_ref, opacity=0.75), name="""Text containing external references (match)""",
                             text=text_references, hoverinfo='text'))
    fig.add_trace(go.Scatter(x=[9999], y=[9999], mode='markers',
                             marker=dict(color="turquoise", opacity=0.75), name='Text containing external references (no match)',
                             text=text_references, hoverinfo='text'))
    fig.add_trace(go.Scatter(x=tsne_embeddings_sent[:, 0], y=tsne_embeddings_sent[:, 1], mode='markers',
                             marker=dict(color=colors_sent, opacity=0.75), name='Unwanted text (match)',
                             text=text_sentences, hoverinfo='text'))
    fig.add_trace(go.Scatter(x=[9999], y=[9999], mode='markers',
                             marker=dict(color="pink", opacity=0.75), name='Unwanted text (no match)',
                             text=text_references, hoverinfo='text'))
    # Add the query vector point
    fig.add_trace(go.Scatter(x=[query_emb[0]], y=[query_emb[1]], mode='markers',
                             marker=dict(color='green', size=10), name=f'Input: {query_text}',
                             text=f'Input: {query_text}', hoverinfo='text'))

    # Set the layout for the figure
    fig.update_layout(title=diagram_title,
                      xaxis=dict(title='TSNE-1'), yaxis=dict(title='TSNE-2'),
                      xaxis_range=[-50, 50], yaxis_range=[-50, 50])

    # Show the figure
    fig.show()


def plot_diagram(doc_id, nr):
    diagram_title = f"Visualization on document {nr} embeddings"
    file_embedding_ref = f"vectors/{doc_id}_references"
    file_embedding_sent = f"vectors/{doc_id}_sentences"

    text_references = data_handler.read_text_file(f"data/{doc_id}_references")
    text_sentences = data_handler.read_text_file(f"data/{doc_id}_sentences")

    embedding_references = data_handler.load_embedding(f"{file_embedding_ref}.pkl")
    embedding_sentences = data_handler.load_embedding(f"{file_embedding_sent}.pkl")

    # Ensure embedding_ref is a numpy array
    embedding_ref = np.array(embedding_references)
    if embedding_ref.ndim > 2:
        embedding_ref = embedding_ref.reshape(len(embedding_ref), -1)

    # Ensure embedding_sent is a numpy array
    embedding_sent = np.array(embedding_sentences)
    if embedding_sent.ndim > 2:
        embedding_sent = embedding_sent.reshape(len(embedding_sent), -1)

    # Now, combine the flattened arrays for t-SNE
    combined_embeddings = np.vstack((embedding_ref, embedding_sent))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    combined_tsne_embeddings = tsne.fit_transform(combined_embeddings)

    # Separate the transformed embeddings
    tsne_embeddings_1 = combined_tsne_embeddings[:len(embedding_ref)]
    tsne_embeddings_2 = combined_tsne_embeddings[len(embedding_ref):]

    # Create a Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=tsne_embeddings_1[:, 0], y=tsne_embeddings_1[:, 1], mode='markers',
                             marker=dict(color="blue", opacity=0.5), name='Text containing external references',
                             text=text_references, hoverinfo='text'))
    fig.add_trace(go.Scatter(x=tsne_embeddings_2[:, 0], y=tsne_embeddings_2[:, 1], mode='markers',
                             marker=dict(color="red", opacity=0.5), name='Unwanted text',
                             text=text_sentences, hoverinfo='text'))

    # Set the layout for the figure
    fig.update_layout(title=diagram_title,
                      xaxis=dict(title='TSNE-1'), yaxis=dict(title='TSNE-2'),
                      xaxis_range=[-50, 50], yaxis_range=[-50, 50])

    # Show the figure
    fig.show()
