# promptQuest

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [How It Works](#how-it-works)
  - [Data Source](#1-data-source)
  - [Preprocessing](#2-preprocessing)
  - [Topic Modeling](#3-topic-modeling)
  - [LLM-based Topic Interpretation](#4-llm-based-topic-interpretation)
  - [Trend Analysis](#5-trend-analysis)
  - [Streamlit Interface](#6-streamlit-interface)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#1-clone-the-repository)
  - [Install Dependencies](#2-install-dependencies)
  - [Set Up Environment](#3-set-up-environment)
- [Usage](#usage)
- [Core Components](#core-components)
  - [`app.py`](#apppy)
  - [`topicmodelling_dev.py`](#topicmodelling_devpy)
  - [`preprocessor.py`](#preprocessorpy)
  - [`cloud_config.py`](#cloud_configpy)
- [Prompt Engineering](#prompt-engineering)
- [Error Handling & Troubleshooting](#error-handling--troubleshooting)
- [Configuration](#configuration)
- [Performance and Cost Caveats](#performance-and-cost-caveats)
- [Cloud Resource Requirements](#cloud-resource-requirements)
- [Example: Topic Extraction Pipeline](#example-topic-extraction-pipeline)
- [Dependencies](#dependencies)

---

## Overview

**promptQuest** is a Streamlit-based analytics dashboard for exploring and analyzing chat data stored in Azure Cosmos DB. It leverages classical NLP (topic modeling) and LLMs (via Azure OpenAI) to analyze chat titles and trends.  

---

## Features

- **Interactive Chat History:** View and filter prior chat interactions by date, quarter, or entry count.
- **Topic Modeling:** Automatically extract and summarize key topics and trends from large batches of chat titles using NMF (Non-negative Matrix Factorization).
- **LLM-powered Analytics:** Use Azure OpenAI to interpret topic clusters and provide human-readable topic names and summaries.
- **Trend Analysis:** Generate simple trend reports on chat activity.
- **Custom Filtering:** Flexible sidebar to filter chat data by monthly, quarterly, custom date range, or number of entries.
- **User-friendly Analytics View:** Visualize quarterly topic analyses and trends.  
  _Note: Output is currently plain text—no charts or graphical KPIs are rendered._
- **Preprocessing Pipeline:** Text cleaning and stopword removal for topic extraction (see details below).

---

## Directory Structure

```
.
├── app.py                # Main Streamlit application
├── topicmodelling_dev.py # Topic modeling and LLM topic interpretation utilities
├── preprocessor.py       # Text preprocessing functions
├── cloud_config.py       # Azure and Cosmos DB configuration
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variable file
├── test_app.py           # (If exists) Example test script
```

---

## How It Works

### 1. Data Source

- Connects to Azure Cosmos DB using credentials from environment variables (see `.env`).
- Fetches chat entries (e.g., question/response titles) from the configured container/database.

### 2. Preprocessing

- Cleans and preprocesses text using NLTK: strips punctuation and removes stopwords.

### 3. Topic Modeling

- Uses `TfidfVectorizer` to vectorize chat titles.
- Applies NMF to identify clusters of keywords representing topics.
- Assembles structured topic data (top keywords per topic, scores, weighted terms).

### 4. LLM-based Topic Interpretation

- Sends extracted topic keywords to Azure OpenAI (GPT-4 or similar).
- Receives a JSON-formatted summary: human-readable topic labels and descriptions.

### 5. Trend Analysis

- Uses LLM for high-level trend summaries from recent chat data.

### 6. Streamlit Interface

- Sidebar controls: choose date range, quarter, or entry count for analysis.
- Main views:
  - **Chat View:** Explore and interact with chat data.
  - **Analytics View:** See quarterly topic breakdowns and trend reports.

---

## Installation

### Prerequisites

- Python 3.8+
- Azure subscription with:
  - Cosmos DB (see [Cloud Resource Requirements](#cloud-resource-requirements) for API type)
  - Azure OpenAI with deployed models (see below)
- `.env` file with required credentials (see below)

### 1. Clone the Repository

```bash
git clone https://github.com/yukeshwarp/promptQuest.git
cd promptQuest
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file in the project root with the following variables (see `.env.example`):

```env
DB_ENDPOINT=your_cosmos_endpoint
DB_KEY=your_cosmos_db_key
DB_NAME=your_db_name
DB_CONTAINER_NAME=your_container_name

LLM_ENDPOINT=your_azure_openai_endpoint
LLM_KEY=your_azure_openai_key
```

---

## Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

- Use the sidebar to load/filter chat data.
- Switch between "Chat View" and "Analytics View" for different perspectives.
- Ask questions about the data in chat view; the LLM will generate responses based on chat history and extracted topics.

---

## Core Components

### `app.py`
- Orchestrates UI, handles user inputs, and coordinates data queries and analysis.
- Manages session state for chat and analytics views.
- Handles querying Cosmos DB and passing data to topic modeling and LLM modules.
- **Note:** Date/offset filtering only; no search or graphical visualizations.

### `topicmodelling_dev.py`
- `extract_topics_from_text`: runs NMF topic modeling and returns structured results.
- `interpret_topics_with_llm`: sends topic clusters to the LLM and parses/returns JSON topic summaries.
- Contains large prompt templates for LLM (see [Prompt Engineering](#prompt-engineering)).

### `preprocessor.py`
- Text cleaning: strips punctuation, removes stopwords.

### `cloud_config.py`
- Loads all cloud credentials and instantiates the Cosmos DB and Azure OpenAI clients.

---

## Prompt Engineering

- **Prompt Objectives:** The code uses custom prompts to instruct the LLM to label topic clusters and generate summaries of chat data.
- **Expected Response Schema:** JSON objects containing topic labels, descriptions, and summaries.
- **Prompt Adaptation:** You can edit prompt templates in `topicmodelling_dev.py` or `app.py` to experiment with schema or objectives.
- **Token & Rate Limit Considerations:** LLM calls are made in loops (e.g., per quarter), which may increase both latency and token costs. See [Performance and Cost Caveats](#performance-and-cost-caveats).

---

## Error Handling & Troubleshooting

Common pitfalls include:

- **Missing NLTK Corpora:**  
  If you see a `LookupError` for stopwords, run:
  ```python
  import nltk
  nltk.download('stopwords')
  ```
- **Invalid Endpoint URLs:**  
  Double-check your `.env` values.
- **401/403 Authorization Errors:**  
  Ensure keys and endpoint URLs are current and valid.
- **Azure Quota or Rate Limits:**  
  LLM calls may fail if rate-limited or if quota is exhausted.
- **Cosmos DB Indexing:**  
  If you plan to add full-text search, ensure your Cosmos DB indexing policy supports it (see [Cloud Resource Requirements](#cloud-resource-requirements)).

---

## Configuration

- **Model Names:**  
  LLM deployment/model names (e.g., `gpt-4.1`, `model-router`) are currently hard-coded in the source (see `cloud_config.py` or LLM call sites).  
  _To change the model, edit the relevant variable or function argument._
- **Environment Variables:**  
  All secrets are loaded from `.env` (see [Installation](#installation)).

---

## Performance and Cost Caveats

- **Looped LLM Calls:**  
  Analytics and topic labeling may call the LLM once per quarter or topic; this can result in high token usage and slow response times.
- **Cost Warnings:**  
  Each LLM call incurs cost; review your OpenAI usage and set limits as needed.

---

## Cloud Resource Requirements

- **Cosmos DB:**  
  - API type: _Core (SQL)_ recommended.  
  - Indexing: If you wish to add full-text search, ensure the indexing policy supports it (see [docs](https://learn.microsoft.com/en-us/azure/cosmos-db/sql/index-policy)).
- **Azure OpenAI:**  
  - You must deploy the required models (e.g., GPT-4) and specify their deployment names.
  - Update your `.env` and code to reference the correct endpoints and deployment names.

---

## Example: Topic Extraction Pipeline

1. Fetch N chat titles from Cosmos DB.
2. Preprocess titles (strip punctuation, remove stopwords).
3. Run NMF to extract clusters of keywords per topic.
4. Send keyword clusters to LLM for human-readable topic labeling.
5. Display results in Streamlit dashboard (plain text only).

---

## Dependencies

- `streamlit` - UI framework
- `scikit-learn` - Topic modeling (NMF, TF-IDF)
- `openai` - Azure OpenAI API client
- `azure-cosmos` - Cosmos DB API client
- `nltk` - Text preprocessing (stopwords)
- `python-dotenv` - Load environment variables

See `requirements.txt` for full list.
