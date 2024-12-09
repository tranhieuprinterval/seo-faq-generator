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

def generate_answers_for_user_questions(user_questions, topic, context, guiding_instructions, api_key):
    openai.api_key = api_key
    questions_list = "\n".join([f"- {q}" for q in user_questions])
    prompt = f"""
You are an expert FAQ generation assistant. Given a topic, optional guiding instructions, and context, produce answers to the user-provided questions in the same "People also ask" format.

For each question from the user, output:
People also ask
<the user question>

<an accurate, helpful, formal answer tied to the context>

Topic: {topic}
Guiding Instructions: {guiding_instructions}
Context:
\"\"\"{context}\"\"\"

User Questions:
{questions_list}

Generate answers for each user question now.
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"system","content":"You are a helpful assistant for generating Q&As."},
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

def parse_generated_qa(content):
    pattern = r"People also ask\s*(.*?)\n(.*?)(?=People also ask|$)"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    parsed_faqs = []
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        parsed_faqs.append((question, answer))
    return parsed_faqs

def format_all_faqs(faqs, user_faqs):
    # Combine all Q&As into a single string for easy copying
    output = []
    for q, a in faqs:
        output.append(f"People also ask\n{q}\n\n{a}\n")
    for q, a in user_faqs:
        output.append(f"People also ask\n{q}\n\n{a}\n")
    return "\n".join(output).strip()

############################################
# Streamlit App
############################################

st.title("Google-Style Q&A Generator")

st.write("""
**How to Use This Tool:**
1. Enter your OpenAI API key.
2. Provide a topic that you want Q&As for.
3. Optionally, enter guiding instructions that influence the style or focus of the Q&As.
4. Optionally, upload an HTML file or provide a URL to use as a context source. The tool will use this context to generate more relevant answers.
5. Specify how many AI-generated Q&As you want and how many user-provided Q&As you want answered.
6. Enter your own specific questions in the "User Questions" box, one question per line.
7. Click "Generate Q&As".
8. Once generated, you can refine individual Q&As using the "Fine-tune" button.
9. When satisfied, click "Copy All Q&As" to get a combined version of all Q&As for easy copying into a Word document.

**Author:** Brandon Lazovic
""")

api_key = st.text_input("OpenAI API key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

uploaded_file = st.file_uploader("Upload an HTML file (optional):", type=["html","htm"])
url = st.text_input("Or enter a URL (optional):")
topic = st.text_area("Topic:", "")
guiding_instructions = st.text_area("Guiding Instructions (optional):", "")
num_ai_faqs = st.number_input("Number of Q&As (generated by AI):", min_value=1, max_value=20, value=5)

user_questions_text = st.text_area("User Questions (optional):", "Enter each question on a new line.")
user_questions = [q.strip() for q in user_questions_text.split("\n") if q.strip()]

num_user_faqs = st.number_input("Number of User Q&As to Generate:", min_value=0, max_value=20, value=len(user_questions))

if "faqs" not in st.session_state:
    st.session_state["faqs"] = []
if "user_faqs" not in st.session_state:
    st.session_state["user_faqs"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""
if "combined_faqs" not in st.session_state:
    st.session_state["combined_faqs"] = ""

generate_button = st.button("Generate Q&As")

if generate_button:
    if not topic.strip() and not (uploaded_file or url.strip()):
        st.error("Please provide a topic or upload/enter a URL for context.")
        st.stop()

    html_content = ""
    if uploaded_file is not None:
        html_content = uploaded_file.read().decode("utf-8", errors="ignore")
    elif url.strip():
        html_content = fetch_html_from_url(url.strip())

    text_content = get_text_content_from_html(html_content) if html_content else ""

    with st.spinner("Generating embeddings..."):
        embeddings = []
        if text_content:
            try:
                embeddings = get_embeddings(text_content, api_key)
            except Exception as e:
                st.warning(f"Could not generate embeddings: {e}")
                embeddings = []

    if topic and embeddings:
        with st.spinner("Retrieving context..."):
            context = retrieve_context(topic, embeddings, api_key)
    else:
        context = text_content[:2000] if text_content else ""

    st.session_state["context"] = context

    # Generate AI-created Q&As
    parsed_faqs = []
    if num_ai_faqs > 0:
        with st.spinner("Generating AI-created Q&As..."):
            try:
                faqs = generate_faqs(topic, context, num_ai_faqs, guiding_instructions, api_key)
            except Exception as e:
                st.error(f"Error generating Q&As: {e}")
                st.stop()
        parsed_faqs = parse_generated_qa(faqs)
        st.session_state["faqs"] = parsed_faqs
    else:
        st.session_state["faqs"] = []

    # Generate answers for user-provided questions (limited to num_user_faqs)
    user_parsed_faqs = []
    if user_questions and num_user_faqs > 0:
        truncated_user_questions = user_questions[:num_user_faqs]
        with st.spinner("Generating answers for user-provided questions..."):
            try:
                user_faqs_content = generate_answers_for_user_questions(truncated_user_questions, topic, context, guiding_instructions, api_key)
                user_parsed_faqs = parse_generated_qa(user_faqs_content)
            except Exception as e:
                st.error(f"Error generating answers for user questions: {e}")

    st.session_state["user_faqs"] = user_parsed_faqs

    if not parsed_faqs and not user_parsed_faqs:
        st.error("No Q&As found. Try adjusting your inputs.")
    else:
        st.success("Q&As generated successfully!")
        # Prepare combined Q&As for copying
        st.session_state["combined_faqs"] = format_all_faqs(st.session_state["faqs"], st.session_state["user_faqs"])

# Display Q&As if available
if st.session_state["faqs"]:
    st.subheader("AI-Generated Q&As")
    for i, (q, a) in enumerate(st.session_state["faqs"]):
        with st.expander(f"Q: {q}"):
            st.write(a)
            if st.button(f"Fine-tune Q&A {i+1}", key=f"fine_tune_{i}"):
                with st.spinner("Refining Q&A..."):
                    try:
                        refined = refine_faq(q, a, topic, st.session_state["context"], guiding_instructions, api_key)
                        ref_match = re.search(r"People also ask\s*(.*?)\n(.*)", refined, flags=re.DOTALL)
                        if ref_match:
                            new_q = ref_match.group(1).strip()
                            new_a = ref_match.group(2).strip()
                            st.session_state["faqs"][i] = (new_q, new_a)
                            st.write("Refined Q&A:")
                            st.write(f"Q: {new_q}")
                            st.write(new_a)
                            # Update combined_faqs after refinement
                            st.session_state["combined_faqs"] = format_all_faqs(st.session_state["faqs"], st.session_state["user_faqs"])
                        else:
                            st.error("Could not parse refined Q&A. Please try again.")
                    except Exception as e:
                        st.error(f"Error refining Q&A: {e}")

if st.session_state["user_faqs"]:
    st.subheader("User-Provided Q&As")
    for i, (q, a) in enumerate(st.session_state["user_faqs"]):
        with st.expander(f"Q: {q}"):
            st.write(a)
            if st.button(f"Fine-tune User Q&A {i+1}", key=f"user_fine_tune_{i}"):
                with st.spinner("Refining user Q&A..."):
                    try:
                        refined = refine_faq(q, a, topic, st.session_state["context"], guiding_instructions, api_key)
                        ref_match = re.search(r"People also ask\s*(.*?)\n(.*)", refined, flags=re.DOTALL)
                        if ref_match:
                            new_q = ref_match.group(1).strip()
                            new_a = ref_match.group(2).strip()
                            st.session_state["user_faqs"][i] = (new_q, new_a)
                            st.write("Refined Q&A:")
                            st.write(f"Q: {new_q}")
                            st.write(new_a)
                            # Update combined_faqs after refinement
                            st.session_state["combined_faqs"] = format_all_faqs(st.session_state["faqs"], st.session_state["user_faqs"])
                        else:
                            st.error("Could not parse refined Q&A. Please try again.")
                    except Exception as e:
                        st.error(f"Error refining user Q&A: {e}")

# Copy all Q&As section
if st.session_state["faqs"] or st.session_state["user_faqs"]:
    if st.button("Copy All Q&As"):
        st.text_area("All Q&As (Copy and paste into your Word doc):", st.session_state["combined_faqs"], height=300)
