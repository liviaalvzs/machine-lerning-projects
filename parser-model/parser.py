import re
import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> N | Det N | Det Adj N | Adj NP | NP PP
VP -> V | V NP | V NP PP | V PP | VP Conj VP
PP -> P NP
"""


grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Lowercase + remove words without alphabetic chars
    """
    words = sentence.lower().split()
    cleaned = []
    for w in words:
        w = re.sub(r'[^a-z]', '', w)  # remove non aphabetic chars
        if any(c.isalpha() for c in w):
            cleaned.append(w)
    return cleaned

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    """
    chunks = []
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        if not any(desc.label() == "NP" for desc in subtree.subtrees(lambda t: t != subtree)):
            chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
