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
        # We use a spinner rather than a progress bar since we don't have a known endpoint
        emb = openai.Embedding.create(model="text-embedding-ada-002", input=[c])["data"][0]["embedding"]
        embeddings.append((c, emb))
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query, embeddings, api_key):
    openai.api_key = api_key
    query_emb = openai.Embedding.create(model="text-embedding-ada-002", input=[query])["data"][0]["embedding"]
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
    # Adjusting the prompt to create "People also ask" style Q&As
    prompt = f"""
You are an expert FAQ generation assistant. Given a topic, optional guiding instructions, and context, produce {num_faqs} formal, English Q&As.
Format them like Google's "People also ask" section, for example:

People also ask
What are standard window sizes?

<answer here>

Make sure each FAQ starts with "People also ask" on its own line, followed by a question and then a detailed, helpful answer. The answers should be accurate, helpful, and tied to the context and topic.

Topic: {topic}
Guiding Instructions: {guiding_instructions}
Context:
\"\"\"{context}\"\"\"

Generate {num_faqs} Q&As now.
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"system","content":"You are a helpful assistant for generating FAQs."},
                  {"role":"user","content": prompt}],
        temperature=0.7,
        max_tokens=2000
    )
    return completion.choices[0].message.content.strip()

def refine_faq(original_q, original_a, topic, context, guiding_instructions, api_key):
    openai.api_key = api_key
    prompt = f"""
Refine the following Q&A to be clearer, more accurate, helpful, and formal in English without changing its intent. Retain the "People also ask" line before the question.

Topic: {topic}
Guiding Instructions: {guiding_instructions}
Context:
\"\"\"{context}\"\"\"

Original:
People also ask
{original_q}

{original_a}

Refined:
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"system","content":"You refine Q&As to improve clarity and helpfulness."},
                  {"role":"user","content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return completion.choices[0].message.content.strip()


############################################
# Streamlit App
############################################

st.title("Google-Style Q&A Generator")

api_key = st.text_input("OpenAI API key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

uploaded_file = st.file_uploader("Upload an HTML file (optional):", type=["html","htm"])
url = st.text_input("Or enter a URL (optional):")
topic = st.text_area("Topic:", "")
guiding_instructions = st.text_area("Guiding Instructions (optional):", "")
num_faqs = st.number_input("Number of Q&As:", min_value=1, max_value=20, value=5)

# Initialize session state
if "faqs" not in st.session_state:
    st.session_state["faqs"] = []  # store (question, answer) tuples
if "context" not in st.session_state:
    st.session_state["context"] = ""

generate_button = st.button("Generate Q&As")

if generate_button:
    if not topic.strip() and not (uploaded_file or url.strip()):
        st.error("Please provide a topic or content.")
        st.stop()

    html_content = ""
    if uploaded_file is not None:
        html_content = uploaded_file.read().decode("utf-8", errors="ignore")
    elif url.strip():
        html_content = fetch_html_from_url(url.strip())

    text_content = get_text_content_from_html(html_content) if html_content else ""
    
    # Show a progress bar while generating embeddings
    with st.spinner("Generating embeddings, please wait..."):
        embeddings = []
        if text_content:
            try:
                embeddings = get_embeddings(text_content, api_key)
            except Exception as e:
                st.warning(f"Could not generate embeddings: {e}")
                embeddings = []

    if topic and embeddings:
        with st.spinner("Retrieving context, please wait..."):
            context = retrieve_context(topic, embeddings, api_key)
    else:
        context = text_content[:2000] if text_content else ""

    st.session_state["context"] = context

    with st.spinner("Generating Q&As with GPT-4, please wait..."):
        try:
            faqs = generate_faqs(topic, context, num_faqs, guiding_instructions, api_key)
        except Exception as e:
            st.error(f"Error generating Q&As: {e}")
            st.stop()

    # Parse the generated Q&As
    # Look for patterns:
    # People also ask
    # [Question] (line after "People also ask")
    # [Answer] (the following lines until next "People also ask" or end)
    # We'll use a regex that looks for:
    # "People also ask" followed by a question line, then some answer lines until next "People also ask".
    pattern = r"People also ask\s*(.*?)\n(.*?)(?=People also ask|$)"
    matches = re.findall(pattern, faqs, flags=re.DOTALL)

    parsed_faqs = []
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        parsed_faqs.append((question, answer))

    if not parsed_faqs:
        st.error("No Q&As found. Try adjusting your inputs.")
    else:
        st.session_state["faqs"] = parsed_faqs
        st.success("Q&As generated successfully!")

# Display Q&As if available
if st.session_state["faqs"]:
    for i, (q, a) in enumerate(st.session_state["faqs"]):
        with st.expander(f"Q: {q}"):
            st.write(a)
            # Provide a fine-tune button
            if st.button(f"Fine-tune Q&A {i+1}", key=f"fine_tune_{i}"):
                with st.spinner("Refining Q&A, please wait..."):
                    try:
                        refined = refine_faq(q, a, topic, st.session_state["context"], guiding_instructions, api_key)
                        # Parse refined
                        # Expecting format:
                        # People also ask
                        # <question>
                        #
                        # <answer>
                        ref_pattern = r"People also ask\s*(.*?)\n(.*)"
                        ref_match = re.search(ref_pattern, refined, flags=re.DOTALL)
                        if ref_match:
                            new_q = ref_match.group(1).strip()
                            new_a = ref_match.group(2).strip()
                            st.session_state["faqs"][i] = (new_q, new_a)
                            st.write("Refined Q&A:")
                            st.write(f"Q: {new_q}")
                            st.write(new_a)
                        else:
                            st.error("Could not parse refined Q&A. Please try again.")
                    except Exception as e:
                        st.error(f"Error refining Q&A: {e}")
