"""
=============================================================================
NLP Pipeline – Customer Product Feedback System
Module 18: Natural Language Processing | Component B: Practical Implementation
Dataset : Walmart Product Reviews (TSV)
Scenario: Scenario 4 – Customer Product Feedback System
=============================================================================

PIPELINE STAGES:
  1. Dataset Loading & Exploration
  2. Text Pre-processing & Handling
  3. Vectorization & Semantic Representation
  4. Grammatical Analysis & Parsing
  5. Information Extraction (NER + Time/Location)
  6. Language Modelling (N-gram + Entropy + Text Generation)
  7. Data Visualisation

USAGE:
  1. Place your Walmart TSV file in the same directory and name it
     'walmart_reviews.tsv'  (or update DATASET_PATH below).
  2. Run:  python nlp_pipeline.py
  3. All output images are saved to ./output_images/
=============================================================================
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import os, re, math, random, warnings, collections
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
# seaborn removed to avoid missing-dependency failure; use matplotlib directly

# NLTK
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

from nltk.tokenize          import word_tokenize, sent_tokenize
from nltk.corpus            import stopwords, wordnet
from nltk.stem              import PorterStemmer, WordNetLemmatizer
from nltk                   import pos_tag, ne_chunk, Tree
from nltk.util              import ngrams
from nltk.lm                import MLE
from nltk.lm.preprocessing  import padded_everygram_pipeline

# spaCy
import spacy
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise         import cosine_similarity

# Gensim Word2Vec
from gensim.models import Word2Vec

# DCG / parse trees
import nltk.parse as nltkparse

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH  = "walmart_reviews.tsv"
OUTPUT_DIR    = "output_images"
RANDOM_SEED   = 42
SAMPLE_SIZE   = 2000      # use a subset for heavy operations; set None for all

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DIVIDER = "\n" + "=" * 70 + "\n"

def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────
# 2.  LOAD & EXPLORE DATASET
# ─────────────────────────────────────────────
print(DIVIDER + "STAGE 1 – DATASET LOADING & EXPLORATION" + DIVIDER)

# ── 2a. Load TSV ──
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found at '{DATASET_PATH}'.\n"
        "Please download it from Kaggle:\n"
        "https://www.kaggle.com/datasets/promptcloud/walmart-product-review-dataset/data\n"
        "and place it in the same folder as this script, named 'walmart_reviews.tsv'."
    )

df = pd.read_csv(DATASET_PATH, sep="\t", on_bad_lines="skip")
print(f"Raw shape  : {df.shape}")
print(f"Columns    : {list(df.columns)}")
print(df.head(3).to_string())

# ── 2b. Identify review text column (robust to column naming) ──
TEXT_COL   = None
RATING_COL = None
for col in df.columns:
    cl = col.lower()
    if TEXT_COL   is None and any(k in cl for k in ["review_text","review","body","text","comment","description"]):
        TEXT_COL = col
    if RATING_COL is None and any(k in cl for k in ["rating","stars","score","overall"]):
        RATING_COL = col

if TEXT_COL is None:
    TEXT_COL = df.columns[0]
    print(f"[WARN] Could not auto-detect text column; defaulting to '{TEXT_COL}'")
else:
    print(f"Text column   detected: '{TEXT_COL}'")

if RATING_COL:
    print(f"Rating column detected: '{RATING_COL}'")

# ── 2c. Clean & sample ──
df = df.dropna(subset=[TEXT_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str)
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sampled {SAMPLE_SIZE} rows for efficiency.")

print(f"Working set: {len(df)} reviews")

# ── 2d. Basic EDA plot ──
if RATING_COL and df[RATING_COL].notna().sum() > 10:
    plt.figure(figsize=(7, 4))
    df[RATING_COL].value_counts().sort_index().plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Rating Distribution – Walmart Product Reviews")
    plt.xlabel("Rating"); plt.ylabel("Count")
    save_fig("01_rating_distribution.png")


# ─────────────────────────────────────────────
# 3.  TEXT PRE-PROCESSING & HANDLING
# ─────────────────────────────────────────────
print(DIVIDER + "STAGE 2 – TEXT PRE-PROCESSING & HANDLING" + DIVIDER)

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ── 3a. Regex cleaning ──
def clean_text(text: str) -> str:
    """
    Regex-based cleaning tailored for product reviews:
      - Remove URLs
      - Remove HTML tags
      - Remove non-ASCII characters
      - Collapse extra whitespace
      - Lower-case
    """
    text = re.sub(r"http\S+|www\.\S+",         "",  text)   # URLs
    text = re.sub(r"<[^>]+>",                  "",  text)   # HTML
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]",  " ", text)   # special chars
    text = re.sub(r"\s+",                      " ", text).strip()
    return text.lower()

# ── 3b. Tokenization ──
def tokenize(text: str):
    return word_tokenize(text)

# ── 3c. Stemming ──
def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

# ── 3d. Lemmatization ──
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

# ── 3e. Stop-word removal ──
def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and t.isalpha()]

# ── 3f. Sentence segmentation ──
def segment_sentences(text: str):
    return sent_tokenize(text)

# ── 3g. Apply full pipeline ──
print("Applying pre-processing pipeline …")
df["cleaned"]    = df[TEXT_COL].apply(clean_text)
df["tokens"]     = df["cleaned"].apply(tokenize)
df["tokens_sw"]  = df["tokens"].apply(remove_stopwords)   # without stop-words
df["stemmed"]    = df["tokens_sw"].apply(stem_tokens)
df["lemmatized"] = df["tokens_sw"].apply(lemmatize_tokens)
df["sentences"]  = df[TEXT_COL].apply(segment_sentences)

# ── 3h. Show comparison ──
sample_idx = 0
print(f"\nOriginal   : {df[TEXT_COL].iloc[sample_idx][:200]}")
print(f"Cleaned    : {df['cleaned'].iloc[sample_idx][:200]}")
print(f"Tokens     : {df['tokens_sw'].iloc[sample_idx][:15]}")
print(f"Stemmed    : {df['stemmed'].iloc[sample_idx][:15]}")
print(f"Lemmatized : {df['lemmatized'].iloc[sample_idx][:15]}")
print(f"Sentences  : {df['sentences'].iloc[sample_idx][:2]}")

# ── 3i. Stemming vs Lemmatization comparison plot ──
example_words = ["running","better","worse","studies","feet","caring","happily","flies"]
stem_results  = [stemmer.stem(w)    for w in example_words]
lemma_results = [lemmatizer.lemmatize(w) for w in example_words]

comp_df = pd.DataFrame({"Original": example_words,
                         "Stemmed" : stem_results,
                         "Lemmatized": lemma_results})
print("\nStemming vs Lemmatization Comparison:")
print(comp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
tbl = ax.table(cellText=comp_df.values, colLabels=comp_df.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
plt.title("Stemming vs Lemmatization", pad=20)
save_fig("02_stem_vs_lemma.png")

# ── 3j. Top-words bar chart (after stop-word removal) ──
all_tokens = [t for toks in df["lemmatized"] for t in toks]
freq       = collections.Counter(all_tokens).most_common(20)
words_freq, counts_freq = zip(*freq)

plt.figure(figsize=(10, 5))
plt.barh(list(words_freq), list(counts_freq), color="skyblue", edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 20 Words in Customer Reviews (lemmatized, stop-words removed)")
plt.xlabel("Frequency")
plt.ylabel("Word")
save_fig("03_top20_words.png")


# ─────────────────────────────────────────────
# 4.  VECTORIZATION & SEMANTIC REPRESENTATION
# ─────────────────────────────────────────────
print(DIVIDER + "STAGE 3 – VECTORIZATION & SEMANTIC REPRESENTATION" + DIVIDER)

corpus = df["cleaned"].tolist()

# ── 4a. Bag-of-Words ──
print("Building Bag-of-Words model …")
bow_vectorizer = CountVectorizer(max_features=5000, stop_words="english")
bow_matrix     = bow_vectorizer.fit_transform(corpus)
bow_vocab      = bow_vectorizer.get_feature_names_out()
print(f"  BoW matrix shape : {bow_matrix.shape}")

# ── 4b. TF-IDF ──
print("Building TF-IDF model …")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english",
                                   ngram_range=(1, 2))
tfidf_matrix     = tfidf_vectorizer.fit_transform(corpus)
tfidf_vocab      = tfidf_vectorizer.get_feature_names_out()
print(f"  TF-IDF matrix shape : {tfidf_matrix.shape}")

# ── 4c. Top TF-IDF terms ──
mean_tfidf  = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top_tfidf_i = mean_tfidf.argsort()[-20:][::-1]
top_tfidf_t = [tfidf_vocab[i] for i in top_tfidf_i]
top_tfidf_v = mean_tfidf[top_tfidf_i]

plt.figure(figsize=(10, 5))
plt.barh(top_tfidf_t, top_tfidf_v, color="seagreen", edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 20 TF-IDF Terms (mean score across all reviews)")
plt.xlabel("Mean TF-IDF Score")
plt.ylabel("Term")
save_fig("04_tfidf_top20.png")

# ── 4d. Semantic Similarity (cosine) ──
print("\nComputing cosine similarity between first 5 reviews …")
sample_vecs = tfidf_matrix[:5]
sim_matrix  = cosine_similarity(sample_vecs)
plt.figure(figsize=(6, 5))
plt.imshow(sim_matrix, cmap="YlOrRd", aspect="equal")
plt.colorbar()
plt.xticks(range(5), [f"R{i}" for i in range(5)])
plt.yticks(range(5), [f"R{i}" for i in range(5)])
plt.title("Cosine Similarity – First 5 Reviews (TF-IDF)")
plt.xlabel("Review")
plt.ylabel("Review")
save_fig("05_cosine_similarity.png")

# ── 4e. Word2Vec ──
print("\nTraining Word2Vec embeddings …")
w2v_sentences = df["tokens_sw"].tolist()
w2v_model = Word2Vec(sentences=w2v_sentences, vector_size=100,
                     window=5, min_count=2, workers=4, seed=RANDOM_SEED, epochs=10)
w2v_vocab_size = len(w2v_model.wv)
print(f"  Word2Vec vocabulary size : {w2v_vocab_size}")

# Similar words demo
demo_words = ["quality", "price", "product", "delivery", "broken"]
for w in demo_words:
    if w in w2v_model.wv:
        similar = w2v_model.wv.most_similar(w, topn=5)
        print(f"\n  Words similar to '{w}': {[x[0] for x in similar]}")

# ── 4f. Word2Vec similarity heatmap ──
seed_words = [w for w in ["quality","price","product","delivery","fast","broken",
                           "good","bad","love","recommend"] if w in w2v_model.wv]
if len(seed_words) >= 4:
    sim_arr = np.array([[w2v_model.wv.similarity(a, b)
                         for b in seed_words] for a in seed_words])
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_arr, cmap="coolwarm", aspect="equal")
    plt.colorbar()
    plt.xticks(range(len(seed_words)), seed_words, rotation=45, ha="right")
    plt.yticks(range(len(seed_words)), seed_words)
    plt.title("Word2Vec Pairwise Semantic Similarity")
    plt.xlabel("Word")
    plt.ylabel("Word")
    save_fig("06_word2vec_similarity.png")


# ─────────────────────────────────────────────
# 5.  GRAMMATICAL ANALYSIS & PARSING
# ─────────────────────────────────────────────
print(DIVIDER + "STAGE 4 – GRAMMATICAL ANALYSIS & PARSING" + DIVIDER)

# ── 5a. POS tagging ──
print("POS Tagging (sample sentences) …")
sample_sentences = df["sentences"].iloc[0][:3]
for sent in sample_sentences:
    tokens = word_tokenize(sent)
    tags   = pos_tag(tokens)
    print(f"\n  Sentence : {sent[:100]}")
    print(f"  POS tags : {tags[:12]}")

# ── 5b. POS distribution ──
pos_counts = collections.Counter()
for toks in df["tokens"].iloc[:500]:
    for _, tag in pos_tag(toks):
        pos_counts[tag] += 1
top_pos = pos_counts.most_common(12)
plt.figure(figsize=(10, 4))
plt.bar([p[0] for p in top_pos], [p[1] for p in top_pos], color="mediumpurple", edgecolor="black")
plt.title("POS Tag Frequency Distribution (sample of 500 reviews)")
plt.xlabel("POS Tag")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
save_fig("07_pos_distribution.png")

# ── 5c. NLTK NE chunking (shallow parse) ──
print("\nShallow Parsing – Named Entity Chunking (NLTK) …")
ner_sent = "The product was shipped from New York by Amazon within two days."
tokens   = word_tokenize(ner_sent)
tags     = pos_tag(tokens)
tree     = ne_chunk(tags, binary=False)
print(f"  Input  : {ner_sent}")
print(f"  Tree   : {tree}")

# ── 5d. spaCy Dependency Parse ──
print("\nDependency Parsing (spaCy) …")
spacy_doc = nlp_spacy(df["cleaned"].iloc[0][:200])
print("  Token   | Dep      | Head")
print("  " + "-"*40)
for tok in list(spacy_doc)[:15]:
    print(f"  {tok.text:<12} {tok.dep_:<12} {tok.head.text}")

# ── 5e. Definite Clause Grammar (DCG) ──
print("\nDefinite Clause Grammar (DCG) Demo …")
grammar_str = r"""
  S  -> NP VP
  NP -> DT NN | DT JJ NN | PRP
  VP -> VBZ NP | VBZ ADJP | VBD NP
  ADJP -> JJ | RB JJ
  DT -> 'the' | 'a' | 'this'
  NN -> 'product' | 'quality' | 'price' | 'item' | 'delivery'
  JJ -> 'great' | 'bad' | 'excellent' | 'poor' | 'fast'
  RB -> 'very' | 'really'
  VBZ -> 'is' | 'has'
  VBD -> 'received' | 'bought'
  PRP -> 'I' | 'it'
"""
grammar = nltk.CFG.fromstring(grammar_str)
parser  = nltk.ChartParser(grammar)

test_sentences = [
    ["the", "product", "is", "great"],
    ["the", "quality", "is", "very", "bad"],
    ["I", "received", "the", "item"],
]
for ts in test_sentences:
    parses = list(parser.parse(ts))
    if parses:
        print(f"\n  Input  : {' '.join(ts)}")
        print(f"  Parse  : {parses[0]}")
    else:
        print(f"\n  Input  : {' '.join(ts)}  ->  [no parse]")

# ── 5f. WordNet Integration ──
print("\nWordNet Integration …")
wn_words = ["quality", "product", "excellent", "broken"]
for word in wn_words:
    synsets = wordnet.synsets(word)
    if synsets:
        syn = synsets[0]
        print(f"\n  Word       : {word}")
        print(f"  Synset     : {syn.name()}")
        print(f"  Definition : {syn.definition()}")
        print(f"  Examples   : {syn.examples()[:2]}")
        hypernyms = syn.hypernyms()
        if hypernyms:
            print(f"  Hypernyms  : {[h.name() for h in hypernyms[:3]]}")
        synonyms = set(l.name() for s in wordnet.synsets(word) for l in s.lemmas())
        print(f"  Synonyms   : {list(synonyms)[:6]}")


# ─────────────────────────────────────────────
# 6.  INFORMATION EXTRACTION
# ─────────────────────────────────────────────
print(DIVIDER + "STAGE 5 – INFORMATION EXTRACTION" + DIVIDER)

# ── 6a. NER with spaCy ──
print("Named Entity Recognition (spaCy) …")

ner_results = {"PERSON": [], "ORG": [], "GPE": [], "PRODUCT": [],
               "DATE": [], "MONEY": [], "CARDINAL": []}

for text in df["cleaned"].iloc[:300]:
    doc = nlp_spacy(text[:500])
    for ent in doc.ents:
        if ent.label_ in ner_results:
            ner_results[ent.label_].append(ent.text)

print("\n  NER Summary (first 300 reviews):")
for label, items in ner_results.items():
    top = collections.Counter(items).most_common(5)
    print(f"  {label:<10}: {top}")

# ── 6b. NER entity count chart ──
entity_counts = {k: len(v) for k, v in ner_results.items() if v}
plt.figure(figsize=(8, 4))
plt.bar(list(entity_counts.keys()), list(entity_counts.values()), color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"][:len(entity_counts)], edgecolor="black")
plt.title("Named Entity Counts (spaCy, first 300 reviews)")
plt.xlabel("Entity Type")
plt.ylabel("Count")
save_fig("08_ner_entity_counts.png")

# ── 6c. Regex-based Time & Location Extraction ──
print("\nRegex-based Time & Location Extraction …")

TIME_PATTERN   = re.compile(
    r"\b(\d{1,2}\s(?:days?|weeks?|months?|hours?)|\d{4}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s\d{1,2}(?:,\s\d{4})?)\b",
    re.IGNORECASE
)
LOCATION_PATTERN = re.compile(
    r"\b(USA|US|United\s+States|China|India|Canada|UK|"
    r"New\s+York|California|Texas|Florida|Amazon|Walmart)\b",
    re.IGNORECASE
)

time_mentions     = []
location_mentions = []
for text in df[TEXT_COL].iloc[:500]:
    time_mentions.extend(TIME_PATTERN.findall(text))
    location_mentions.extend(LOCATION_PATTERN.findall(text))

print(f"  Time expressions found     : {len(time_mentions)}")
print(f"  Sample times               : {time_mentions[:8]}")
print(f"  Location expressions found : {len(location_mentions)}")
loc_freq = collections.Counter([l.upper() for l in location_mentions]).most_common(10)
print(f"  Top locations              : {loc_freq}")

# ── 6d. Location bar chart ──
if loc_freq:
    locs, lcounts = zip(*loc_freq)
    plt.figure(figsize=(8, 4))
    plt.barh(list(locs), list(lcounts), color="orange", edgecolor="black")
    plt.gca().invert_yaxis()
    plt.title("Top Location Mentions in Reviews (Regex Extraction)")
    plt.xlabel("Count")
    plt.ylabel("Location")
    save_fig("09_location_extraction.png")

# ── 6e. Product/Brand NER demo ──
print("\nProduct/Brand Entity Demo (spaCy) …")
demo_review = ("I ordered this Samsung TV from Walmart and it arrived in 3 days. "
               "The product quality is amazing. Shipped from California.")
doc_demo = nlp_spacy(demo_review)
print(f"  Text    : {demo_review}")
print(f"  Entities: {[(e.text, e.label_) for e in doc_demo.ents]}")


# ─────────────────────────────────────────────
# 7.  LANGUAGE MODELLING
# ─────────────────────────────────────────────
print(DIVIDER + "STAGE 6 – LANGUAGE MODELLING" + DIVIDER)

# ── 7a. Build N-gram corpus ──
print("Building N-gram Language Model (trigram) …")
lm_tokens = [lst for lst in df["tokens_sw"].tolist() if len(lst) >= 3]

N = 3   # trigram
train_data, padded_vocab = padded_everygram_pipeline(N, lm_tokens)
lm = MLE(N)
lm.fit(train_data, padded_vocab)
print(f"  Vocabulary size (LM): {len(lm.vocab)}")

# ── 7b. N-gram frequency demo ──
bigram_counts  = collections.Counter()
trigram_counts = collections.Counter()
for toks in lm_tokens[:500]:
    bigram_counts.update(ngrams(toks, 2))
    trigram_counts.update(ngrams(toks, 3))

print("\n  Top 10 Bigrams:")
for gram, cnt in bigram_counts.most_common(10):
    print(f"    {gram}  ->  {cnt}")

print("\n  Top 10 Trigrams:")
for gram, cnt in trigram_counts.most_common(10):
    print(f"    {gram}  ->  {cnt}")

# ── 7c. N-gram frequency bar chart ──
bg_labels = [" ".join(g) for g, _ in bigram_counts.most_common(15)]
bg_vals   = [c            for _, c in bigram_counts.most_common(15)]
plt.figure(figsize=(10, 5))
plt.barh(bg_labels, bg_vals, color="coral", edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 15 Bigrams in Customer Reviews")
plt.xlabel("Frequency")
plt.ylabel("Bigram")
save_fig("10_bigram_frequency.png")

# ── 7d. Perplexity & Entropy ──
print("\nPerplexity & Entropy Calculation …")

def calc_entropy_perplexity(model, test_sents, n):
    """
    Manual cross-entropy and perplexity calculation.
    H  = -1/N * sum(log2 P(w_i | context))
    PP = 2^H
    """
    log_prob_sum = 0.0
    word_count   = 0
    for sent in test_sents:
        padded = list(nltk.lm.preprocessing.pad_both_ends(sent, n=n))
        grams  = list(ngrams(padded, n))
        for gram in grams:
            context = gram[:-1]
            word    = gram[-1]
            prob    = model.score(word, context)
            if prob > 0:
                log_prob_sum += math.log2(prob)
                word_count   += 1
    if word_count == 0:
        return float("inf"), float("inf")
    entropy    = -log_prob_sum / word_count
    perplexity = 2 ** entropy
    return entropy, perplexity

test_subset = lm_tokens[:100]
entropy, perplexity = calc_entropy_perplexity(lm, test_subset, N)
print(f"  Model Entropy    : {entropy:.4f}  bits/word")
print(f"  Model Perplexity : {perplexity:.2f}")
print("\n  Interpretation:")
print(f"  • Entropy of {entropy:.2f} bits/word indicates the model's uncertainty per word.")
print(f"  • Perplexity of {perplexity:.0f} means the model is as uncertain as choosing")
print(f"    uniformly from ~{int(perplexity)} words at each step.")
print("  • Lower perplexity = better predictive model.")

# ── 7e. Text Generation ──
print("\nText Generation using Trigram Model …")

def generate_text(model, seed_words, num_words=15, max_tries=200):
    """Generate text by sampling from the language model."""
    generated = list(seed_words)
    context   = list(seed_words[-(N-1):]) if len(seed_words) >= N-1 else seed_words[:]
    for _ in range(num_words):
        for attempt in range(max_tries):
            next_word = model.generate(1, text_seed=context, random_seed=random.randint(0,9999))
            if next_word and next_word != "<UNK>" and next_word not in ("<s>","</s>"):
                generated.append(next_word)
                context = (context + [next_word])[-(N-1):]
                break
        else:
            break
    return " ".join(generated)

seed_examples = [("product", "quality"), ("fast", "delivery"), ("great", "value")]
print("\n  Generated text samples:")
for seed in seed_examples:
    if all(s in lm.vocab for s in seed):
        text = generate_text(lm, list(seed), num_words=12)
        print(f"  Seed {seed}: {text}")
    else:
        print(f"  Seed {seed}: (seed words not in vocabulary)")

# ── 7f. Sentiment distribution (simple rule-based) ──
print("\nSimple Sentiment Labelling (rating-based) …")

if RATING_COL and df[RATING_COL].notna().sum() > 10:
    try:
        df["rating_num"] = pd.to_numeric(df[RATING_COL], errors="coerce")
        df["sentiment"]  = df["rating_num"].apply(
            lambda r: "Positive" if r >= 4 else ("Negative" if r <= 2 else "Neutral")
        )
        sent_counts = df["sentiment"].value_counts()
        plt.figure(figsize=(6, 4))
        colors = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}
        sent_counts.plot(kind="bar",
                         color=[colors.get(s, "grey") for s in sent_counts.index],
                         edgecolor="black")
        plt.title("Sentiment Distribution (Rating-based)")
        plt.xlabel("Sentiment"); plt.ylabel("Count")
        plt.xticks(rotation=0)
        save_fig("11_sentiment_distribution.png")
        print(f"  {sent_counts.to_dict()}")
    except Exception as e:
        print(f"  [WARN] Could not compute sentiment: {e}")

# ── 7g. Word frequency over time (review index as proxy) ──
print("\nWord Trend Analysis (review index as time proxy) …")
trend_words = [w for w in ["great","bad","quality","broken","recommend"]
               if w in all_tokens][:4]
if trend_words:
    chunk_size = max(1, len(df) // 10)
    chunks     = [df["lemmatized"].iloc[i:i+chunk_size]
                  for i in range(0, len(df), chunk_size)]
    chunk_freqs = {}
    for w in trend_words:
        chunk_freqs[w] = [sum(w in toks for toks in chunk) / max(1, len(chunk))
                          for chunk in chunks]
    plt.figure(figsize=(10, 4))
    for w in trend_words:
        plt.plot(chunk_freqs[w], marker="o", label=w)
    plt.title("Word Frequency Trends Across Review Batches")
    plt.xlabel("Review Batch"); plt.ylabel("Relative Frequency")
    plt.legend(); plt.grid(alpha=0.3)
    save_fig("12_word_trends.png")


# ─────────────────────────────────────────────
# 8.  SUMMARY
# ─────────────────────────────────────────────
print(DIVIDER + "PIPELINE COMPLETE – SUMMARY" + DIVIDER)
print(f"  Dataset reviewed  : {len(df)} Walmart product reviews")
print(f"  Pre-processing    : Regex clean, tokenize, stop-words, stem, lemmatize")
print(f"  Vectorization     : BoW ({bow_matrix.shape}), TF-IDF ({tfidf_matrix.shape})")
print(f"  Word2Vec vocab    : {w2v_vocab_size} words")
print(f"  NER entities      : {sum(len(v) for v in ner_results.values())} found")
print(f"  Time extractions  : {len(time_mentions)}")
print(f"  Location mentions : {len(location_mentions)}")
print(f"  LM vocabulary     : {len(lm.vocab)}")
print(f"  Entropy           : {entropy:.4f}  |  Perplexity: {perplexity:.2f}")
print(f"\n  Output images saved to : ./{OUTPUT_DIR}/")
print("\n  Images generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    {f}")
print(DIVIDER)
