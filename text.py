"""Text normalization and safe output formatting utilities."""
import re

def clean_text(text):

    if not isinstance(text,str):
        raise TypeError("Text must be a string")

    text=text.casefold()
    # Keep numbers and common technical-token characters. Removing these
    # makes terms such as C++, C#, RFC-822, x86, and .NET unretrievable.
    text=re.sub(r"[^\w\s@#$%&+./:-]"," ",text,flags=re.UNICODE)
    return re.sub(r"\s+"," ",text).strip()



def format_snippet(text,max_chars=200):

    """Return a bounded, single-line preview for interactive output."""

    if not isinstance(text,str):
        raise TypeError("Text must be a string")

    if max_chars<=0:
        raise ValueError("max_chars must be positive")

    snippet=" ".join(text.split())

    if len(snippet)<=max_chars:
        return snippet

    if max_chars<=3:
        return snippet[:max_chars]

    return snippet[:max_chars-3].rstrip()+"..."
