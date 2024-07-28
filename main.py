import streamlit as st
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import  requests
from bs4 import BeautifulSoup
import re


from keybert import KeyBERT
from rouge_score import rouge_scorer


import nltk
nltk.download('punkt')
from nltk import  sent_tokenize



def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Function to download and preprocess Gutenberg books
def download_gutenberg_book(book_id):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-h/{book_id}-h.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    # Basic preprocessing
    text = ' '.join(text.split())
    return text


# Download some Gutenberg books
gutenberg_books = {
    11: download_gutenberg_book(11),  # Alice in Wonderland
    1342: download_gutenberg_book(1342),  # Pride and Prejudice
    1661: download_gutenberg_book(1661)  # The Adventures of Sherlock Holmes
}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def preprocess_book(text):
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.!?]', '', text)
    text = text.lower()

    # Tokenize into sentences
    sentences = sent_tokenize(text)

    # Split into chunks of roughly 1000 tokens
    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if token_count + len(tokens) > 1000:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            token_count = 0
        current_chunk.append(sentence)
        token_count += len(tokens)

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)

    return chunks


if 'downloaded_books' not in st.session_state:
    st.session_state['downloaded_books'] = {}

# Preprocess the downloaded books
preprocessed_books = {book_id: preprocess_book(text) for book_id, text in gutenberg_books.items()}

# Create the corpus
corpus = [chunk for chunks in preprocessed_books.values() for chunk in chunks]





# Initialize models and tokenizer
@st.cache_resource
def initialize_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    keyword_model = KeyBERT()
    return summarizer, tokenizer, model, embed_model, keyword_model


summarizer, tokenizer, model, embed_model, keyword_model = initialize_models()


# Load dataset
@st.cache_data
def load_booksum_dataset():
    return load_dataset("kmfoda/booksum", split="test")


booksum_dataset = load_booksum_dataset()


# Load FAISS index
@st.cache_resource
def load_faiss_index(corpus):
    corpus_embeddings = embed_model.encode(corpus)
    #corpus_embeddings = corpus_embeddings.detach().numpy()  # Convert tensor to numpy array

    print(type(corpus_embeddings))
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)
    return index


index = load_faiss_index(corpus)


def update_faiss_index(new_chunks):
    global index, corpus
    new_embeddings = embed_model.encode(new_chunks)
    #new_embeddings = new_embeddings.cpu().detach().numpy()  # Convert tensor to numpy array
    faiss.normalize_L2(new_embeddings)
    if index is None:
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
    index.add(new_embeddings)
    corpus.extend(new_chunks)
    st.session_state['corpus'] = corpus


# Retrieve passages function
def retrieve_passages(query, top_k=3):
    query_vector = embed_model.encode([query])
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, top_k)
    print(f"Retrieved indexes: {I[0]}")
    return [st.session_state['corpus'][i] for i in I[0]]


# Summarization functions
def generate_summary(text, max_length=150):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length,
                                 min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def rag_summarize_concat(text):
    initial_summary = generate_summary(text)
    retrieved_passages = retrieve_passages(initial_summary)
    combined_text = initial_summary + " " + " ".join(retrieved_passages)
    final_summary = generate_summary(combined_text, max_length=200)
    return final_summary


def rag_summarize_extracted(text):
    initial_summary = generate_summary(text)
    retrieved_passages = retrieve_passages(initial_summary)

    # Extract keywords from retrieved passages
    keywords = []
    for passage in retrieved_passages:
        keywords.extend([kw for kw, _ in keyword_model.extract_keywords(passage, top_n=5, stop_words='english')])

    # Add unique keywords to the initial summary
    unique_keywords = list(set(keywords))
    augmented_text = initial_summary + " Additional key information: " + ", ".join(unique_keywords)

    final_summary = generate_summary(augmented_text, max_length=200)
    return final_summary


def rag_summarize_guided(text):
    initial_summary = generate_summary(text)
    retrieved_passages = retrieve_passages(initial_summary)

    # Create a prompt that includes the initial summary and guidance from retrieved passages
    prompt = f"Summarize the following text, focusing on these key points: {initial_summary}\n\nAdditional context:\n"
    for i, passage in enumerate(retrieved_passages, 1):
        prompt += f"{i}. {passage}\n"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to('cpu')
    guided_summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0,
                                        num_beams=4, early_stopping=True)
    return tokenizer.decode(guided_summary_ids[0], skip_special_tokens=True)


# Streamlit app layout
st.title("Book Summarization and Evaluation")

# Sidebar for book selection
st.sidebar.title("Select a Book by BID")
book_ids = [book['bid'] for book in booksum_dataset]
selected_book_id = st.sidebar.selectbox("Choose a book BID", book_ids)

# Get the book text from the dataset
book_text = [book['chapter'] for book in booksum_dataset if book['bid'] == selected_book_id][0]
reference_summary = [book['summary'] for book in booksum_dataset if book['bid'] == selected_book_id][0]

st.subheader(f"Text of the book with BID: {selected_book_id}")
st.write(book_text[:200] + "...")  # Display only the first 2000 characters for brevity

# Option to download the book from Project Gutenberg and update FAISS index
if st.button("Download and Add Book to Index"):
    with st.spinner("Downloading and processing book..."):
        new_book_text = download_gutenberg_book(selected_book_id)
        new_chunks = preprocess_book(new_book_text)
        update_faiss_index(new_chunks)
        st.session_state['downloaded_books'][selected_book_id] = new_book_text
        st.success("Book added to FAISS index.")

# Generate summaries
if st.button("Generate Summaries"):
    with st.spinner("Generating summaries..."):
        baseline_summary = generate_summary(book_text)
        st.subheader("Baseline Summary")
        st.write(baseline_summary)
        baseline_rouge_scores = calculate_rouge(reference_summary, baseline_summary)
        st.write("Baseline ROUGE Scores:")
        st.json(baseline_rouge_scores)
        print(gutenberg_books.keys())
        if selected_book_id in st.session_state['downloaded_books']:
            rag_concat_summary = rag_summarize_concat(book_text)
            rag_extracted_summary = rag_summarize_extracted(book_text)
            rag_guided_summary = rag_summarize_guided(book_text)

            st.subheader("RAG Concatenation Summary")
            st.write(rag_concat_summary)

            rag_concat_rouge_scores = calculate_rouge(reference_summary, rag_concat_summary)
            st.write("RAG Concatenation ROUGE Scores:")
            st.json(rag_concat_rouge_scores)

            st.subheader("RAG Extracted Summary")
            st.write(rag_extracted_summary)
            rag_extracted_rouge_scores = calculate_rouge(reference_summary, rag_extracted_summary)
            st.write("RAG Extracted ROUGE Scores:")
            st.json(rag_extracted_rouge_scores)

            st.subheader("RAG Guided Summary")
            st.write(rag_guided_summary)

            rag_guided_rouge_scores = calculate_rouge(reference_summary, rag_guided_summary)
            st.write("RAG Guided ROUGE Scores:")
            st.json(rag_guided_rouge_scores)
        else:
            st.warning("Download and add the book to the index to generate RAG summaries.")

st.subheader("Rate Summaries")
ratings = {}
for summary_type in ["Baseline", "RAG Concatenation", "RAG Extracted", "RAG Guided"]:
    st.write(f"Rate the {summary_type} Summary")
    ratings[summary_type] = st.slider(f"{summary_type} Effectiveness", 0, 5, 3)

st.write("Ratings:", ratings)