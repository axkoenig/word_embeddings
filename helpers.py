import os

from tika import parser

def pdf2txt(path, doc_name):
    """Converts a pdf into a txt file.
    """
    full_path = os.path.join(path, doc_name) + ".pdf"
    save_path = os.path.join(path, doc_name) + ".txt"
    parsed = parser.from_file(full_path)
    content = parsed["content"]
    
    with open(save_path, "w") as f:
        f.write(content)

