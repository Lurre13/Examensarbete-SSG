import support_files.handle_pdf as pdf_handler
import support_files.OpenaiApi as OpenAI_api
import support_files.handle_data as data_handler
import support_files.handle_vectors as vector_handler


def extract_data_from_source(path_to_input_pdf, similarity_query, llm_prompt):
    openAI = OpenAI_api.OpenaiApi()
    embeddings = []
    sections = pdf_handler.extract_and_divide_text_pdf(path_to_input_pdf)
    for section in sections:
        embeddings.append(openAI.vectorize_string(section))

    text = vector_handler.extract_text_with_cosine_similarity(embeddings, similarity_query, 0.45, sections)
    openAI.extract_data(text, llm_prompt)


doc_id = "2001_5"
input_pdf = f"data/{doc_id}.pdf"
query = "Enligt standard SS EN ISO 5555"
threshold = 0.45

# Different prompt levels to see how the LLM react
prompt_level_1 = """Du är en hjälpsam assistent som har i uppdrag att hitta externa referenser från meningar. Svara
                    enbart med en lista på de hittade externa referenserna, om det skulle vara så att du inte hittade
                     någon svara med ''"""

prompt_level_2 = """Du är en hjälpsam assistent som har i uppdrag att hitta externa referenser från meningar.
                  Dessa externa referenser är formaterade som sifferkod eller standarder som vanligtvis erkänns i
                   tekniska och vetenskapliga dokument, till exempel 'SS', 'EN' eller 'ISO' eller en kombination av
                    dem, men det finns även fler. Svara enbart med en lista på de hittade externa referenserna, om det
                     skulle vara så att du inte hittade någon svara med ''"""

prompt_level_3 = """Du är en hjälpsam assistent som har i uppdrag att hitta externa referenser från meningar.
                  Dessa externa referenser är formaterade som sifferkod eller standarder som vanligtvis erkänns i
                   tekniska och vetenskapliga dokument, till exempel 'SS', 'EN' eller 'ISO' eller en kombination av
                    dem, men det finns även fler. Så din uppgift är följande: hitta alla externa referenser. Det kan
                     vara så att det också finns vilken del av standarden som den refererar till och kan göras på två
                      sätt, ena är genom ett '-x' där 'x' är en siffra, och det andra är genom att skriva 'del x' där
                       'x' är en siffra. Om du hittar en referens som skrivs med 'del' sättet så omvandla det till '-x'
                        och inkludera det i referensen. Det kan också vara så att referensen innehåller årtal på
                         följande sätt ':xxxx' där 'xxxx' är ett årtal. Inkludera detta i referensen om det finns. Sen
                          ta bort referenser till allmänna publikationer, företagsinterna dokument från 'SSG' eller
                          några hyperlänkar. Nästa steg är att ta bort referenser som du har hittat flera gånger, detta
                           inkluderar då hittade referenser som har samma kod men del nummer och årtal får vara olika,
                            ta bort duplikat genom att exkludera de referenser som innehåller minst information. Svara
                             enbart med en lista på de hittade externa referenserna, om det skulle vara så att du inte
                              hittade någon svara med ''"""


print("\nPrompt level 1")
extract_data_from_source(input_pdf, query, prompt_level_1)
print("\nPrompt level 2")
extract_data_from_source(input_pdf, query, prompt_level_2)
print("\nPrompt level 3")
extract_data_from_source(input_pdf, query, prompt_level_3)
