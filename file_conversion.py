from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# Path to data folder
DATA_FOLDER = Path("data")

# Find all PDFs and PowerPoints
pdf_files = list(DATA_FOLDER.glob("*.pdf"))

print("PDF Files:")
print(pdf_files)

all_text = ""

for pdf in pdf_files:
    print(f"\nReading PDF: {pdf.name}")

    loader = PyPDFLoader(str(pdf))
    documents = loader.load()

    print(f"Pages: {len(documents)}")

    for page in documents:
        all_text += page.page_content
        all_text += "\n\n"

# Create docs folder if it doesn't exist
docs_folder = Path("docs")
docs_folder.mkdir(exist_ok=True)

# Save extracted text
output_file = docs_folder / "all_notes.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write(all_text)

print(f"\n✅ Text saved to: {output_file}")
