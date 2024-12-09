import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
import re
import numpy as np

############################################
# Helper Functions
############################################

def get_text_content_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_html_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return ""

def get_embeddings(text, api_key):
    openai.api_key = api_key
    chunks = []
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    embeddings = []
    for c in chunks:
        # Pass the text as a list to the input parameter
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[c]
        )
        emb = response["data"][0]["embedding"]
        embeddings.append((c, emb))
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query, embeddings, api_key):
    openai.api_key = api_key
    query_emb = openai.Embedding.create(model="text-embedding-ada-002", input=query)["data"][0]["embedding"]
    best_score = -1
    best_chunk = ""
    for txt, emb in embeddings:
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_chunk = txt
    return best_chunk

def generate_faqs(topic, context, num_faqs, guiding_instructions, api_key):
    openai.api_key = api_key
    prompt = f"""
You are an expert FAQ generation assistant. Given a topic, optional guiding instructions, and context, produce {num_faqs} formal, English FAQs relevant to the topic. Each FAQ: "Q: ...\nA: ...". The answers should be accurate, helpful, and tied to the context.

Topic: {topic}
Guiding Instructions: {guiding_instructions}
Context:
\"\"\"{context}\"\"\"

Generate {num_faqs} FAQs now.
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"system","content":"You are a helpful assistant for generating FAQs."},
                  {"role":"user","content": prompt}],
        temperature=0.7,
        max_tokens=3000
    )
    return completion.choices[0].message.content.strip()

def refine_faq(original_q, original_a, topic, context, guiding_instructions, api_key):
    openai.api_key = api_key
    prompt = f"""
Refine the following FAQ to be clearer, more accurate, helpful, and formal in English without changing its intent:

Topic: {topic}
Guiding Instructions: {guiding_instructions}
Context:
\"\"\"{context}\"\"\"

Original:
Q: {original_q}
A: {original_a}

Refined:
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"system","content":"You refine FAQs to improve clarity and helpfulness."},
                  {"role":"user","content": prompt}],
        temperature=0.7,
        max_tokens=3000
    )
    return completion.choices[0].message.content.strip()

############################################
# Streamlit App
############################################

st.title("FAQ Generator from Web Page")

api_key = st.text_input("OpenAI API key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

uploaded_file = st.file_uploader("Upload an HTML file (optional):", type=["html","htm"])
url = st.text_input("Or enter a URL (optional):")
topic = st.text_area("FAQ Topic:", "")
guiding_instructions = st.text_area("Guiding Instructions (optional):", "")
num_faqs = st.number_input("Number of FAQs:", min_value=1, max_value=20, value=5)

if st.button("Generate FAQs"):
    html_content = ""
    if uploaded_file is not None:
        html_content = uploaded_file.read().decode("utf-8", errors="ignore")
    elif url.strip():
        html_content = fetch_html_from_url(url.strip())

    text_content = get_text_content_from_html(html_content) if html_content else ""
    embeddings = get_embeddings(text_content, api_key) if text_content else []

    if topic and embeddings:
        context = retrieve_context(topic, embeddings, api_key)
    else:
        context = text_content[:2000] if text_content else ""

    faqs = generate_faqs(topic, context, num_faqs, guiding_instructions, api_key)
    pattern = r"(Q:\s*.*?\nA:\s*.*?(?=\nQ:|$))"
    matches = re.findall(pattern, faqs, flags=re.DOTALL)

    if not matches:
        st.error("No FAQs found. Try adjusting your inputs.")
    else:
        for i, m in enumerate(matches):
            q_match = re.search(r"Q:\s*(.*)", m)
            a_match = re.search(r"A:\s*(.*)", m)
            q = q_match.group(1).strip() if q_match else ""
            a = a_match.group(1).strip() if a_match else ""

            with st.expander(f"Q: {q}"):
                st.write(f"A: {a}")
                if st.button(f"Fine-tune FAQ {i+1}"):
                    refined = refine_faq(q, a, topic, context, guiding_instructions, api_key)
                    rq = re.search(r"Q:\s*(.*)", refined)
                    ra = re.search(r"A:\s*(.*)", refined)
                    rq = rq.group(1).strip() if rq else q
                    ra = ra.group(1).strip() if ra else a
                    st.write("Refined FAQ:")
                    st.write(f"Q: {rq}")
                    st.write(f"A: {ra}")
