#==========================Pre-processing Module===========================
# This module defines the clean_text_content function, which is responsible for cleaning plain text content while preserving paragraph structure. 
# The function takes a string of text as input and performs several cleaning steps, including normalizing newlines, removing invisible characters, converting tabs to spaces, 
# removing trailing spaces at line ends, collapsing repeated spaces inside lines, reducing multiple blank lines to a maximum of two, and stripping overall leading and trailing whitespace. 
# The cleaned text is returned as a string, which can then be used for further processing in the RAG agent, such as splitting into chunks and generating embeddings. 
# This pre-processing step is crucial for ensuring that the text data is in a clean and consistent format, which can improve the performance of the retrieval and response generation processes in the RAG agent.

import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List


def clean_text_content(text: str) -> str:
    """
    Clean plain text content while preserving paragraph structure.
    Suitable for TXT files before chunking.
    """
    if not text:
        return ""

    # 1. Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove BOM / zero-width / invisible characters
    invisible_chars = ["\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"]
    for ch in invisible_chars:
        text = text.replace(ch, "")

    # 3. Convert tabs to spaces
    text = text.replace("\t", " ")

    # 4. Remove trailing spaces at line ends
    text = re.sub(r"[ \t]+\n", "\n", text)

    # 5. Collapse repeated spaces inside lines
    text = re.sub(r"[ ]{2,}", " ", text)

    # 6. Reduce 3+ blank lines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Strip overall leading/trailing whitespace
    text = text.strip()

    return text


def extract_title_from_text(text: str) -> Optional[str]:
    """
    Use the first non-empty line as a simple title candidate.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return None


def is_heading_candidate(line: str) -> bool:
    """
    Very light heuristic for heading detection in plain text files.
    Avoids being too aggressive.
    """
    line = line.strip()
    if not line:
        return False

    # Too short or too long usually not useful as heading
    if len(line) < 3 or len(line) > 80:
        return False

    # Skip normal sentences ending with strong punctuation
    if line.endswith((".", "?", "!", ";", ":")):
        return False

    # Match numbered headings like "1 Introduction" or "2.1 Data Source"
    if re.match(r"^\d+(\.\d+)*\s+[A-Za-z].*$", line):
        return True

    # Match all-caps headings like "OVERVIEW"
    if line.isupper() and len(line.split()) <= 8:
        return True

    # Match title-like short lines
    words = line.split()
    if len(words) <= 8 and all(word[:1].isupper() for word in words if word.isalpha()):
        return True

    return False


def extract_headings(text: str, max_headings: int = 20) -> List[str]:
    """
    Extract possible headings from plain text using a simple heuristic.
    """
    headings = []
    seen = set()

    for line in text.split("\n"):
        candidate = line.strip()
        if is_heading_candidate(candidate) and candidate not in seen:
            headings.append(candidate)
            seen.add(candidate)

        if len(headings) >= max_headings:
            break

    return headings


def build_text_metadata(
    raw_text: str,
    cleaned_text: str,
    source_name: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build metadata for a TXT/plain-text document.
    """
    lines = cleaned_text.split("\n") if cleaned_text else []
    non_empty_lines = [line for line in lines if line.strip()]
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]

    title = extract_title_from_text(cleaned_text)
    headings = extract_headings(cleaned_text)

    metadata = {
        "source_type": "text",
        "source_name": source_name or (Path(source_path).name if source_path else "unknown.txt"),
        "source_path": source_path,
        "file_extension": Path(source_path).suffix.lower() if source_path else ".txt",
        "title": title,
        "headings": headings,
        "raw_char_count": len(raw_text),
        "clean_char_count": len(cleaned_text),
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty_lines),
        "paragraph_count": len(paragraphs),
        "word_count": len(re.findall(r"\b\w+\b", cleaned_text)),
        "sha1": hashlib.sha1(cleaned_text.encode("utf-8")).hexdigest(),
    }

    return metadata


def process_text_document(
    text: str,
    source_name: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Clean text and return a normalized document record.
    """
    cleaned_text = clean_text_content(text)
    metadata = build_text_metadata(
        raw_text=text,
        cleaned_text=cleaned_text,
        source_name=source_name,
        source_path=source_path,
    )

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }



if __name__ == "__main__":
    sample_text = """
   INTRODUCTION


This is a sample text file.    
It has extra spaces.

1 Overview
This is the overview section.


DATA SOURCES

The system uses text, pdf, and html inputs.
"""
    doc = process_text_document(sample_text, source_name="sample.txt")

    print(doc["text"])
    print(doc["metadata"])