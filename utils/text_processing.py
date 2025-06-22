import re

def clean_text(text: str) -> str:
    """Basic text cleaning function"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'[^\w\s.,;:!?]', '', text)  # Remove special chars
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitting"""
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

def tokenize_with_offsets(text: str):
    """Tokenize text with character offsets"""
    tokens = []
    start = 0
    for word in text.split():
        end = start + len(word)
        tokens.append((word, start, end))
        start = end + 1  # +1 for the space
    return tokens