import streamlit as st
import pandas as pd
import plotly.express as px
from azure.cosmos import CosmosClient
from topicmodelling_dev import extract_topics_from_text
from cloud_config import *
from cloud_config import redis_url
import time
from datetime import datetime, timedelta
import re
from wordcloud import STOPWORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import dcc
import concurrent.futures
from celery import Celery
import random
import redis
import json


# Enhanced Page Configuration
st.set_page_config(
    page_title="promptQuest",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Celery setup
app = Celery(
    'summarization_tasks',
    broker=redis_url, 
    backend=redis_url,
    include=['summarization_tasks']  
)


app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    broker_transport_options={
        'visibility_timeout': 3600,  # Task visibility timeout (in seconds)
        'max_connections': 100,  # Number of Redis connections
    },
)

st.header("_promptQuest Analytics_")

# Initialize Session State with Default Values
def initialize_session_state():
    default_states = {
        "messages": [],
        "text_content": "",
        "topics": [],
        "topic_data": None,
        "chat_titles": [],
        "current_page": "Dashboard",
        "time_filter": "All Time",
        "refresh_data": True
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

def trend_analysis(mode=st.session_state["time_filter"]):
    prompt = ""
    text_content = st.session_state["text_content"]
    # st.write(text_content)
    topics = st.session_state["topics"]
    if mode == "All Time":
        prompt = f"""Analyse the trend over monthly under 40 words, with bullets of key points (For display in the dashboard of application) for the database and return the output in a redable analysis format
                    Database of application usage by users: {text_content}
                    Highlighted topics: {topics}"""
    elif mode == "Quaterly":
        prompt = f"""Analyse the trend over monthly under 40 words, with bullets of key points (For display in the dashboard of application) for the database and return the output in a redable analysis format
                    Database of application usage by users: {text_content}
                    Highlighted topics: {topics}"""
    else:
        prompt = f"""Analyse the trend over monthly under 40 words, with bullets of key points (For display in the dashboard of application) for the database and return the output in a redable analysis format
                    Database of application usage by users: {text_content}
                    Highlighted topics: {topics}"""
    
    response = llmclient.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in analysing trend in application usage based on application log database."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content

@app.task
def summarize(text):
    max_tries = 5
    base_delay = 1
    for attempt in range(max_tries):
        try:
            response = llmclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant expert in summarizing."},
                    {"role": "user", "content": f"Summarize the following chat in less than 25 words with understanding of intent in the chat: {text}"}
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Summarizing DB resulted in error after try{attempt}. \nError Details: {e}")
            if attempt<max_tries:
                delay = base_delay*(2**attempt) + random.uniform(1,0)
                time.sleep(delay)
            else:
                raise e

# Redis Setup
redis_client = redis.StrictRedis.from_url(redis_url, decode_responses=True)

# Cache processed data
def cache_processed_data(chat_id, processed_data):
    redis_client.set(chat_id, json.dumps(processed_data))  # Cache for 24 hours

# Retrive available data
def get_cached_data(chat_id):
    cached_data = redis_client.get(chat_id)
    return json.loads(cached_data) if cached_data else None

def fetch_chat_titles(limit=250, time_filter="All Time"):
    new_chat_ids = []
    cached_results = []
    
    query = "SELECT c.id, c.TimeStamp, c.AssistantName, c.ChatTitle, c.category FROM c ORDER BY c.TimeStamp DESC OFFSET 0 LIMIT @limit"
    params = [{"name": "@limit", "value": limit}]

    client = CosmosClient(ENDPOINT, KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
    
    for item in items:
        chat_id = item["id"]
        cached_data = get_cached_data(chat_id)
        if cached_data:
            cached_results.append(cached_data)
        else:
            new_chat_ids.append(item)

    if new_chat_ids:
        new_titles = [item["ChatTitle"][:100] if item["AssistantName"] == "Summarize" else item["ChatTitle"] for item in new_chat_ids]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            summaries = list(executor.map(summarize, new_titles))

        for item, summary in zip(new_chat_ids, summaries):
            processed_data = {
                "title": summary[:25] or item.get("ChatTitle", "Untitled"),
                "timestamp": item.get("TimeStamp"),
                "assistant": item.get("AssistantName", "Unknown"),
            }
            cache_processed_data(item["id"], processed_data)
            cached_results.append(processed_data)

    return cached_results

def analyze_topics(chat_titles):
    if not chat_titles:
        st.warning("No chat titles available for topic analysis.")
        return "", []
    
    try:
        titles = [item.get("title", "") for item in chat_titles]
        text_content = " ".join(titles)
        
        if not text_content.strip():
            st.warning("Empty text content after processing chat titles.")
            return "", []
        
        try:
            topics = extract_topics_from_text(text_content)
        except Exception as topic_error:
            st.error(f"Error extracting topics: {topic_error}")
            return text_content, []
        
        topic_data = topics.split(",") if topics else []
        
        if not topic_data:
            st.warning("No topics extracted from chat titles.")
        
        return text_content, topic_data
    
    except Exception as e:
        st.error(f"Unexpected error in analyze_topics: {e}")
        return "", []

# Modify the main data fetching function
@st.cache_data(ttl=3600)
def fetch_and_process_data(limit, time_filter):
    try:
        # Fetch chat titles with comprehensive error handling
        chat_titles = fetch_chat_titles(limit, time_filter)
        
        if not chat_titles:
            st.warning("No chat titles retrieved. Check your data source and filters.")
            return [], "", []
        
        # Analyze topics
        text_content, topic_data = analyze_topics(chat_titles)
        
        return chat_titles, text_content, topic_data
    
    except Exception as e:
        st.error(f"Comprehensive data fetching error: {e}")
        return [], "", []

with st.sidebar:
    st.markdown("### 🔍 Data Filters")
    
    # Time Period Selection
    time_options = ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Quarterly", "Monthly", "Half-Yearly", "Yearly", "All Time"]
    selected_time = st.selectbox("Time Period", time_options, index=time_options.index(st.session_state["time_filter"]))
    
    # Record Limit Slider
    limit = st.slider("Number of Records", min_value=150, max_value=12000, value=2000, step=50)
    
    # Conditional Quarterly Selection
    if selected_time == "Quarterly":
        year_options = [str(year) for year in range(2020, datetime.now().year + 1)]
        selected_year = st.selectbox("Select Year", year_options, index=year_options.index(str(datetime.now().year)))
        selected_quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
        st.session_state["selected_year"] = selected_year
        st.session_state["selected_quarter"] = selected_quarter
    
    # Refresh Button
    if st.button("🔄 Refresh Data", key="main_refresh"):
        st.session_state["refresh_data"] = True

prompt = st.chat_input("Ask a question")
if st.session_state["refresh_data"]:
    with st.spinner("Fetching and processing data..."):
        chat_titles, text_content, topic_data = fetch_and_process_data(limit, selected_time)
        
        # Update Session State
        st.session_state["chat_titles"] = chat_titles   
        st.session_state["text_content"] = text_content
        st.session_state["topics"] = topic_data
        st.session_state["refresh_data"] = False

# Trend Analysis
st.markdown('<div class="sub-header">Trend Overview</div>', unsafe_allow_html=True)
st.markdown(trend_analysis())

st.markdown('<div class="main-header">Interactive Chat Insights</div>', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.markdown('</div>', unsafe_allow_html=True)

# User input for questions
if prompt:
    text_content = st.session_state["text_content"]
    topics = st.session_state["topics"]
    bot_response = ""
    if text_content and topics:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            time.sleep(1) 
            topic_names = [t for t in topics]
            response_stream = llmclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert product analyst who analyses software products based on the user statistics from user database."},
                    {"role": "user", "content": f"""
                        Answer the user's prompt based on the following data from the database. 
                        The database contains usage history of user questions and AI responses from an AI-assisted chatbot interface, specifically used for legal advice.
                        
                        User Database: {text_content}
                        Highlighted Topics: {topics}
                        
                        ---
                        Prompt: {prompt}
                        
                        ---
                        Intelligently analyze the user's intent in the prompt and provide an insightful answer, utilizing relevant data and context from the database.
                        """
                    }
                ],
                temperature=0.7,
                stream=True,
            )
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_stream:
                if chunk.choices:
                    bot_response += chunk.choices[0].delta.content or ""
                    message_placeholder.markdown(bot_response)
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
        
    else:
        st.warning("Please fetch and analyze topics first.")