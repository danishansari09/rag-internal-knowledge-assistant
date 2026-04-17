import re
def remove_space_and_newlines_for_text(text):
   # 1. Normalize newlines (convert \r\n to \n)
    text = text.replace('\r\n', '\n')
    
    # 2. Replace 3 or more newlines with exactly 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 3. Remove weird invisible characters (e.g., zero-width spaces)
    text = text.replace('\u200b', '')
    
    # 4. Optional: Remove extra spaces between words
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()