import PyPDF2
from nltk.tokenize import sent_tokenize


def read_pdf(filepath):
    """
    Read content of a PDF file

    :param filepath: Filepath for file to extract text from
    :return: A string containing all the text
    """
    with open(filepath, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Check if the PDF has at least one page
        if len(pdf_reader.pages) > 0:
            text_list = ""
            for page in pdf_reader.pages:
                # Extract text from each page
                text_list += page.extract_text()

            return text_list
        else:
            return ""


def extract_and_divide_text_pdf(filepath):
    """
    Extracts sentences from a text

    :param filepath: Filepath for file to extract text from
    :return: A list containing all the sentences
    """
    pdf_text = read_pdf(filepath)
    sentences = sent_tokenize(pdf_text, language="swedish")

    return sentences


def extract_pages_by_group(input_pdf_path, base_output_pdf_path, group_size=5):
    with open(input_pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)

        # Iterate through the PDF in steps of 'group_size'
        for start_page in range(0, total_pages, group_size):
            writer = PyPDF2.PdfWriter()
            # Determine the last page in the current group
            end_page = min(start_page + group_size, total_pages)

            # Add pages from start_page to end_page-1
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # Generate a unique name for each output PDF
            output_pdf_path = f"{base_output_pdf_path}_pages_{start_page + 1}_to_{end_page}.pdf"

            # Write out the new PDF
            with open(output_pdf_path, 'wb') as output_file:
                writer.write(output_file)


def extract_pages(input_pdf_path, output_pdf_path, pages_to_keep):
    # Open the source PDF
    with open(input_pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        writer = PyPDF2.PdfWriter()

        # Add the pages specified in pages_to_keep
        for page_num in pages_to_keep:
            # PyPDF2 uses 0-based indexing for pages
            writer.add_page(reader.pages[page_num])

        # Write out the new PDF
        with open(output_pdf_path, 'wb') as output_file:
            writer.write(output_file)
