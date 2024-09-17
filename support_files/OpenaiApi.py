import openai
from openai import OpenAI
import os
from support_files import handle_data as data_handler


class OpenaiApi:
    # Access the key from the environment variable
    def __init__(self):
        openai_api_key = os.getenv('OPENAI_KEY')

        if openai_api_key is None:
            raise ValueError("The OPENAI_KEY environment variable is not set.")

        openai.api_key = openai_api_key

    def extract_data(self, input_text, prompt):
        client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0.2,
            messages=[
                {"role": "system",
                 "content": prompt

                 },
                {"role": "user", "content": f"{input_text}"},
            ]
        )

        print(completion.choices[0].message.content)

    def vectorize_string(self, text):
        openai.api_key = os.getenv('OPENAI_KEY')
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        # Print the resulting embedding vector
        return response.data[0].embedding

    def vectorize_list(self, sentences_list, references_list, doc_id):
        embeddings_sent = []
        embeddings_ref = []

        for sent in sentences_list:
            embeddings_sent.append(self.vectorize_string(sent))

        for sent in references_list:
            embeddings_ref.append(self.vectorize_string(sent))

        #data_handler.save_embedding(f"{doc_id}_sentences_new", embeddings_sent)
        #data_handler.save_embedding(f"{doc_id}_references_new", embeddings_ref)

        data_handler.save_embedding(f"vectors/{doc_id}_sentences", embeddings_sent)
        data_handler.save_embedding(f"vectors/{doc_id}_references", embeddings_ref)
