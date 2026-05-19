# parser.py
# splits the free-text Q1 answer into separate objection points.
# people format their answers in lots of different ways (1., *, -, blank lines...)
# so we just try each pattern until one of them gives us 2+ pieces.

# regex stuff I keep forgetting -> https://docs.python.org/3/howto/regex.html
# really useful for testing patterns -> https://regex101.com/
# beginner walkthrough -> https://realpython.com/regex-python/

import re
import pandas as pd


def parse_q1(text):
    # bail out on empty / non-string cells
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()

    # some people start with stuff like "I object because:" or "My concerns are:"
    # we just chop that off so it doesn't end up as point #1
    text = re.sub(r'^(?:I (?:object|refuse)|My concerns are).*?:\s*', '', text, flags=re.IGNORECASE)

    #  numbered: "1." "1)" "#1." "1.)" etc.
    # this is the most common ChatGPT-ish format people paste in
    if re.search(r'(?:^|\n)\s*#?\d+(?:\.\)|[\.\)])\s?', text):
        parts = re.split(r'(?:^|\s)\s*#?\d+(?:\.\)|[\.\)])\s*', text)
        parts = [s.strip().rstrip('.') for s in parts if s.strip() and len(s.strip()) > 3]
        if len(parts) >= 2:
            return parts

    #  asterisk bullets:  * thing
    if re.search(r'(?:^|\n)\s*\*', text):
        parts = re.split(r'(?:^|\n)\s*\*\s*', text)
        parts = [s.strip() for s in parts if s.strip()]
        if len(parts) >= 2:
            return parts

    #  hash bullets:  # thing
    # the (?=[A-Za-z]) lookahead avoids matching "#1." which is handled above
    # lookahead reference: https://docs.python.org/3/library/re.html#regular-expression-syntax
    if re.search(r'(?:^|\n)\s*#\s?(?=[A-Za-z])', text):
        parts = re.split(r'(?:^|\n)\s*#\s*', text)
        parts = [s.strip() for s in parts if s.strip()]
        if len(parts) >= 2:
            return parts

    #  dash bullets:  - thing
    if re.search(r'(?:^|\n)\s*-\s?(?=\S)', text):
        parts = re.split(r'(?:^|\n)\s*-\s*', text)
        parts = [s.strip() for s in parts if s.strip()]
        if len(parts) >= 2:
            return parts

    #  blank line between paragraphs
    if '\n\n' in text:
        parts = [s.strip() for s in text.split('\n\n') if s.strip()]
        if len(parts) >= 2:
            return parts

    #  single newline
    if '\n' in text:
        parts = [s.strip() for s in text.split('\n') if s.strip()]
        if len(parts) >= 2:
            return parts

    #  semicolons
    # also strip "and " off the front of the next chunk so we don't get "and X"
    if ';' in text:
        parts = re.split(r'\s*;\s*', text)
        parts = [re.sub(r'^and\s+', '', s).strip() for s in parts if s.strip()]
        if len(parts) >= 2:
            return parts

    #  periods (only if each chunk is short-ish so we don't shred a paragraph)
    # TODO: 150 chars is a guess, might want to tune this later
    if text.count('.') >= 2:
        parts = re.split(r'\.\s+', text)
        parts = [s.strip().rstrip('.') for s in parts if s.strip()]
        if len(parts) >= 2 and all(len(p) < 150 for p in parts):
            return parts

    #  commas (last resort, same length)
    # see this SO answer about splitting on multiple delimiters:
    # https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
    if ',' in text:
        parts = [s.strip() for s in text.split(',') if s.strip()]
        if len(parts) >= 2 and all(len(p) < 150 for p in parts):
            return parts

    # nothing matched -> treat the whole answer as one point
    return [text]


def parse_participants(responses_df):
    # responses_df has 2 cols: Participant #, Q1
    rows = []
    for _, row in responses_df.iterrows():
        pid = row["Participant #"]
        points = parse_q1(row["Q1"])
        for i, pt in enumerate(points, 1):
            rows.append({"Participant #": pid, "Point Index": i, "Point": pt})
    return pd.DataFrame(rows)


def load_responses(path):
    # only need the first two columns (id + Q1)
    df = pd.read_excel(path)
    df = df.iloc[:, :2]
    df.columns = ["Participant #", "Q1"]
    df = df.dropna(subset=["Q1"])
    return df
