# walmart-review-nlp-project
A Natural Language Processing pipeline project for text preprocessing, TF-IDF analysis, similarity analysis, POS tagging, named entity recognition, and visualization using Python.

NLP Review Analysis Pipeline
Project Overview

This project presents a complete Natural Language Processing (NLP) pipeline developed using Python. The system performs preprocessing, text analysis, feature extraction, similarity analysis, and visualization on customer review data.

The project uses Walmart customer review data to demonstrate several important NLP techniques including tokenization, stemming, lemmatization, TF-IDF vectorization, cosine similarity, Word2Vec similarity, POS tagging, Named Entity Recognition (NER), and n-gram analysis.

The purpose of this project is to demonstrate practical implementation of NLP techniques in a structured and automated pipeline.

Features
Text preprocessing
Tokenization
Stopword removal
Stemming and lemmatization
TF-IDF feature extraction
Cosine similarity analysis
Word2Vec similarity analysis
Part-of-speech tagging
Named Entity Recognition (NER)
Bigram frequency analysis
Data visualization using graphs and charts
Technologies Used
Python
Pandas
NumPy
NLTK
spaCy
Scikit-learn
Gensim
Matplotlib
Seaborn
Dataset

The project uses Walmart customer review data stored in TSV format.

Dataset file:
              Data/walmart_reviews.tsv

Project Structure

files/
│
├── Data/
│   └── walmart_reviews.tsv
│
├── output_images/
│   ├── 02_stem_vs_lemma.png
│   ├── 03_top20_words.png
│   ├── 04_tfidf_top20.png
│   ├── 05_cosine_similarity.png
│   ├── 06_word2vec_similarity.png
│   ├── 07_pos_distribution.png
│   ├── 08_ner_entity_counts.png
│   ├── 09_location_extraction.png
│   ├── 10_bigram_frequency.png
│   └── 12_word_trends.png
│
├── nlp_pipeline.py
├── requirements.txt
├── README.md
└── run_log.txt
