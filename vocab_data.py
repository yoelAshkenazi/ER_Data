"""
The scripts in this file are used to create a VocabData graph, which uses a multilingual graph to set the entities,
(blue edges)
and uses BERT to get embeddings for all words in a given language, then create a knn graph using cosine-similarity
(red edges).

Overall:
The vertices are words in different languages (each language is a vertex type),
The blue edges are between words that translate to each other (the same entity),
The red edges are between words that are similar in meaning (knn graph in each language).
"""

# Imports
from PyMultiDictionary import MultiDictionary, DICT_EDUCALINGO
from typing import List, Tuple
from sklearn.utils import shuffle
from tqdm import tqdm
import torch
import networkx as nx
from transformers import BertModel, BertTokenizer
from itertools import product
import matplotlib.pyplot as plt


# load the words
with open("random_words.txt", "r") as f:
    DATA = f.readlines()
DATA = [x.strip() for x in DATA]


# Base word lists for different sizes.
def gen_random_word_list(size) -> List[str]:
    """
    Create a list of random words from the english dictionary.
    :param size: int, the size of the word list.
    :return: list of str, the word list.
    """

    return shuffle(DATA)[:size]


# Create a list of tuples of translations for a given list of english words and list of languages.
def gen_translation_list(words: List[str], languages: List[str]) -> List[tuple]:
    """
    Create a list of tuples of translations for a given list of english words and list of languages.
    :param words: list of str, the list of english words.
    :param languages: list of str, the list of languages.
    :return: list of tuple, the list of translations.
    """

    # Create a MultiDictionary object.
    md = MultiDictionary(*words)

    language_to_idx = {lang: i for i, lang in enumerate(languages)}

    # Create a list of tuples of translations.
    vocab_data = [[] for _ in range(len(words))]
    for i in tqdm(range(len(words))):
        # Get the translations for the word.
        translations = md.translate(word=words[i], lang='en', dictionary=DICT_EDUCALINGO)

        # Filter out the translations that are not in the given languages.
        relevant_translations = [x for x in translations if x[0] in language_to_idx]

        # Order the translations by language. (The same order as the given languages)
        relevant_translations = sorted(relevant_translations, key=lambda x: language_to_idx[x[0]])
        relevant_translations = [x[1] for x in relevant_translations]  # Keep only the words.

        # Add the original word at the beginning of the list.
        relevant_translations = [words[i]] + relevant_translations

        # Add the translations to the list of translations.
        vocab_data[i] = tuple(relevant_translations)

    return vocab_data


# Get a list of words in a given language, and return a list of embedding vectors from BERT.
def gen_embedding_list(words: List[str]):
    """
    Get a list of words in a given language, and return a list of embedding vectors from BERT.
    :param words: list of str, the list of words.
    :return:
    """

    # Load the BERT model and tokenizer.
    model = BertModel.from_pretrained(f'bert-base-multilingual-cased')
    tokenizer = BertTokenizer.from_pretrained(f'bert-base-multilingual-cased')

    embeddings = []
    # Get the embedding vector for each word (in the given language). All the embeddings should have the same size.
    for word in words:
        # Tokenize the word.
        input_ids = tokenizer.encode(word, return_tensors='pt')

        # Get the embedding vector.
        with torch.no_grad():
            output = model(input_ids)
            current_output = torch.Tensor(output[0].mean(dim=1).numpy()[0])
            embeddings.append(current_output)

    return embeddings


# Given a list of embedding vectors, create a knn graph using cosine-similarity.
def gen_knn_graph(words, k=5) -> List[tuple]:
    """
    Given a list of embedding vectors, create a knn graph using cosine-similarity.
    :param words: list of str, the list of words.
    :param k: int, the number of nearest neighbors.
    :return: list of tuples (int, int), the list of edges in the knn graph.
    """

    embeddings = gen_embedding_list(words)

    word_to_idx = {word: i for i, word in enumerate(words)}

    for i, e in enumerate(embeddings):  # make sure all embeddings are Tensors
        if not isinstance(e, torch.Tensor):
            raise ValueError(f"Embedding at index {i} is not a torch.Tensor.")

    # Create the adjacency matrix of the knn graph.
    adj = torch.zeros(len(embeddings), len(embeddings))

    # Calculate the cosine-similarity between each pair of embeddings.
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            # Calculate the cosine-similarity.
            similarity = (torch.dot(embeddings[i], embeddings[j]) /
                          (torch.norm(embeddings[i]) * torch.norm(embeddings[j])))

            # Add the similarity to the adjacency matrix.
            adj[i, j] = similarity

    # Create the knn graph.
    knn_graph = torch.zeros(len(embeddings), len(embeddings))
    for i in range(len(embeddings)):
        # Get the indices of the k-nearest neighbors.
        _, indices = adj[i].topk(k + 1)  # +1 to remove the self-loop.

        # Set the knn graph.
        knn_graph[i, indices] = 1

    # Get the edges of the knn graph.
    _edges = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if knn_graph[i, j] == 1:
                _edges.append((i, j))

    return _edges, word_to_idx


# Build the VocabData graph: blue edges between words that translate to each other,
# red edges between words that are similar in meaning.
def build_vocab_graph(size, languages, k=5) -> Tuple[List[tuple], List[tuple]]:
    """
    Build the VocabData graph: blue edges between words that translate to each other,
    red edges between words that are similar in meaning.
    :param size: int, the size of the word list.
    :param languages: list of str, the list of languages.
    :param k: int, the number of nearest neighbors.
    :return: tuple of list of tuple, the list of blue edges and the list of red edges.
    """

    # Generate a list of random words.
    list_of_words = gen_random_word_list(size)

    # Generate a list of translations.
    translations_gen = gen_translation_list(list_of_words, languages)
    languages = ['en'] + languages  # add english to the languages
    # Create the graph.
    G = nx.Graph()

    lang_to_shape = {'en': 'o', 'fr': 's', 'de': 'd', 'es': 'p', 'it': 'h', 'pt': '8'}

    # Add the vertices to the graph.
    for line in translations_gen:
        for idx in range(len(line)):
            G.add_node(line[idx], lang=languages[idx], shape=lang_to_shape[languages[idx]])

    # Generate the blue edges.
    for i in range(len(translations_gen)):  # iterate over the translations
        for w1, w2 in product(translations_gen[i], translations_gen[i]):
            if G.nodes()[w1]['lang'] != G.nodes()[w2]['lang']:  # if the words are in different languages
                G.add_edge(w1, w2, color='blue')

    # Generate the red edges.
    for lang in languages:
        lang_words = [x for x in G.nodes() if G.nodes()[x]['lang'] == lang]
        knn_edges, word_to_idx = gen_knn_graph(lang_words, k=k)
        for i, j in knn_edges:  # iterate over the edges
            G.add_edge(lang_words[i], lang_words[j], color='red')

    print(G)
    print(G.nodes)
    print(G.edges)

    # placeholder - save G

    return G


def draw_graph(G, languages_):
    """
    Draw the VocabData graph.
    :param languages_:  list of str, the list of languages.
    :param G: nx.Graph, the graph to draw.
    """

    # Draw the graph.
    pos = nx.spring_layout(G)
    # Draw the vertices according to their language (shape).
    for lang in languages_:
        vertices = [x for x in G.nodes() if G.nodes()[x]['lang'] == lang]
        shape = G.nodes()[vertices[0]]['shape']
        nx.draw_networkx_nodes(G, pos, nodelist=vertices, node_shape=shape, node_color='r', alpha=0.8)

    # Draw the edges according to their color.
    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, alpha=0.5)
    plt.show()


#%% test
G_ = build_vocab_graph(100, ['fr', 'de', 'es'], k=5)
langs_ = ['en', 'fr', 'de', 'es']
draw_graph(G_, langs_)
