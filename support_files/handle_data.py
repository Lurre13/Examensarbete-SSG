import pickle
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()  # Read the entire file into a string
        sections = text.split("\n\n")  # Split the text into sections
        return sections


def write_text_file(file_path, text_list):
    with open(file=file_path, mode="w", encoding='utf-8') as file:
        for sent in text_list:
            file.write(sent + "\n\n")


def save_embedding(output_file, embeddings):
    # Saving vectors to a file
    with open(output_file + ".pkl", 'wb') as f:
        pickle.dump(embeddings, f)


def load_embedding(input_file):
    # Loading vectors from a file
    with open(input_file, 'rb') as f:
        embeddings = pickle.load(f)
        return embeddings


def remove_stopwords(sentence_list):
    # Get the list of stopwords in English
    stop_words = set(stopwords.words('swedish'))
    stop_words.discard("din")
    filtered_sentence_list = []

    for sentence in sentence_list:
        # Tokenize the text into words
        words = word_tokenize(sentence)

        # Remove stopwords from the tokenized words list
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join words to form the sentence again
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentence_list.append(filtered_sentence)

    return filtered_sentence_list

def load_embeddings_and_text(doc_id):
    file_embedding_ref = f"vectors/{doc_id}_references"
    file_embedding_sent = f"vectors/{doc_id}_sentences"
    file_text_ref = f"data/{doc_id}_references"
    file_text_sente = f"data/{doc_id}_sentences"

    text_references = read_text_file(f"{file_text_ref}")
    text_sentences = read_text_file(f"{file_text_sente}")

    embedding_references = load_embedding(f"{file_embedding_ref}.pkl")
    embedding_sentences = load_embedding(f"{file_embedding_sent}.pkl")

    return embedding_sentences, embedding_references, text_sentences, text_references
