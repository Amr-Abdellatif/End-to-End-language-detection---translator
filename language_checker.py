import re

def detect_language(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')  # Arabic Unicode range
    english_pattern = re.compile(r'[a-zA-Z]+')  # English alphabet

    if arabic_pattern.search(text):
        return "Arabic"
    elif english_pattern.search(text):
        return "English"
    else:
        return "Unknown"
