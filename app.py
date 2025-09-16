import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="SentimentStream", layout="wide")

# Load dataset
df = pd.read_csv("employee_sentiment_dataset.csv")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./bert_sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("./bert_sentiment_model")
    return tokenizer, model

tokenizer, model = load_model()

# Precompute TF-IDF vectorizer for chatbot
@st.cache_resource
def get_vectorizer_and_matrix(messages):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(messages)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df['message'].fillna(""))

# Sentiment Classifier
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return "Positive" if predicted_class == 1 else "Negative"

# Chatbot Message Search
def search_messages(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return [(df['message'].iloc[i], similarities[i]) for i in top_indices if df['message'].iloc[i]]

# Sidebar navigation
st.sidebar.title("🔀 Navigation")
selection = st.sidebar.radio("Go to", ["🏠 Landing Page", "📊 Dashboard", "🤖 Chatbot"])

# === LANDING PAGE ===
if selection == "🏠 Landing Page":
    st.title("🚀 Welcome to SentimentStream")
    st.markdown("""
    **SentimentStream** empowers executives, HR professionals, and operations leaders to make fast, proactive decisions by turning complex employee data into simple, actionable insights using AI.
    """)

    st.subheader("Core Features")
    col1, col2 = st.columns(2)

    with col1:
        st.image("https://via.placeholder.com/300x180", caption="Unified Sentiment Dashboard")
        st.write("Monitor morale, engagement, and communication trends across departments in real time.")

    with col2:
        st.image("https://via.placeholder.com/300x180", caption="Attrition Risk Detection")
        st.write("Use behavioral and feedback signals to predict workforce risks early and accurately.")

    col3, col4 = st.columns(2)

    with col3:
        st.image("https://via.placeholder.com/300x180", caption="Skill Matching Engine")
        st.write("Match employee skills with roles and projects based on resume embeddings and training data.")

    with col4:
        st.image("https://via.placeholder.com/300x180", caption="Natural Language Dashboards")
        st.write("Ask questions like: *'Where are the top burnout risks this month?'* and get immediate answers.")

    st.markdown("---")
    st.subheader("Get In Touch")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success("Thanks! We'll reach out shortly.")

    st.markdown("<hr><center><small>©️ 2025 SentimentStream | All rights reserved.</small></center>", unsafe_allow_html=True)

# === DASHBOARD PAGE ===
elif selection == "📊 Dashboard":
    st.title("📊 Employee Sentiment Dashboard")

    dept_filter = st.sidebar.multiselect("Select Departments", options=df['department'].unique(), default=df['department'].unique())
    project_filter = st.sidebar.multiselect("Select Projects", options=df['project'].unique(), default=df['project'].unique())

    filtered_df = df[df['department'].isin(dept_filter) & df['project'].isin(project_filter)]

    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_pie = px.pie(sentiment_counts, names='sentiment', values='count', title='Sentiment Distribution')
    st.plotly_chart(fig_pie)

    fig_bar = px.bar(filtered_df.groupby('department')['sentiment'].value_counts().unstack().fillna(0),
                     title="Sentiment by Department", barmode='group')
    st.plotly_chart(fig_bar)

    fig_proj = px.bar(filtered_df.groupby('project')['sentiment'].value_counts().unstack().fillna(0),
                      title="Sentiment by Project", barmode='group')
    st.plotly_chart(fig_proj)

    st.subheader("💬 Most Negative Comments")
    most_negative = df[(df['sentiment'] == 'negative') & (df['message'].notnull())]
    most_negative = most_negative[['employee_id', 'department', 'project', 'message']].head(10)
    st.dataframe(most_negative)

    st.subheader("🔍 Try It Yourself: Classify New Message")
    user_input = st.text_area("Enter employee message here:")
    if user_input:
        prediction = get_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{prediction}**")

# === CHATBOT TOOL PAGE ===
elif selection == "🤖 Chatbot":
    st.title("🤖 Ask the Sentiment Bot")
    st.write("Type in a question like:")
    st.code("What do employees think about Project Alpha?")
    st.code("What is the feeling in HR department?")

    query = st.text_input("Ask a question:")
    if query:
        comments = df[df['message'].notnull()]['message'].tolist()
        corpus = [query] + comments
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
        top_indices = cosine_similarities.argsort()[-5:][::-1]
        top_responses = [comments[i] for i in top_indices]

        st.markdown("**Top Relevant Employee Messages:**")
        for i, response in enumerate(top_responses, 1):
            st.write(f"**{i}.** {response}")
