 # Copyright : Petroglyphs NLP Consulting
 # author: Jerome Massot (jerome.massot.78@gmail.com)

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from transformers import pipeline

import streamlit as st
import pandas as pd
import pinecone
import nltk

# download nltk component
nltk.download('punkt')


@st.experimental_singleton
def init_pinecone():
    api_key = st.secrets['API_KEY']
    pinecone.init(api_key=api_key, environment='us-west1-gcp')
    return pinecone.Index('video-search')
    

@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')


@st.experimental_singleton
def load_qa_pipeline():
    return pipeline("question-answering")


def reconstruct_answered_context(query, top_k=3):
 
    # embed the query
    xq = retriever.encode([query]).tolist()
    
    # search the contexts
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    
    # reconstruct the contexts
    returned_sentences = set()
    for context in xc['matches']:
        text_context = context['metadata']['text']
        sentences = sent_tokenize(text_context)
        answer = question_answerer(question=query, context=text_context)['answer']
        for sentence in sentences:
            if sentence.find(answer) > -1:
                returned_sentences.add((sentence, context['metadata']['url'], context['metadata']['index']))
                break

    return returned_sentences

# interface

st.image("./decorations/logo-climate-now.svg", width=100),

st.title("ClimateNow Videos and Podcasts Q&A Engine")
st.subheader("Explore knowledge contained in ClimateNow Video and Podcast channel")

st.info(
    """
    Disclamer: the content shown in this page is the property of Climate Now. 
    Petroglyphs NLP Consulting has only done the content parsing, the indexing and the 
    Semantic Search Engine.
    """
)

# Index and Retriever model setup
index = init_pinecone()
retriever = init_retriever()
question_answerer = load_qa_pipeline()

query = st.text_input("Question:", help="enter your question here")
top_k = st.number_input("Nb of returned context:", 1, 5, help="Ranking contexts from videos")
search = st.button("Search")

if search and query != "":
    returned_sentences = list(reconstruct_answered_context(query, top_k))
    columns = st.columns(len(returned_sentences))
    for i, col in enumerate(columns):
        with col:
            start = int(returned_sentences[i][1].split('t=')[-1][:-1])
            st.video(returned_sentences[i][1], start_time=start)
            st.markdown(f"**Answer**: {returned_sentences[i][0]}")
            st.markdown(f"**Topic**: {returned_sentences[i][2]}")
