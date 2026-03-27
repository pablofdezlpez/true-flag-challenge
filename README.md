# True Flag Chatbot challenge

This is a project created in response to the exercise proposed by True Flag about a RAG agent for claim reviews.

The solution includes the ingestion of a csv file provided by true flag into a chroma database.

The agent will receive a query of a possible fake claim in the form of text or image (or both). With this query the agent will retrieve relevant documents from chroma database in order to determine if the claim is known to be false or not. In addition to retrieving the information, the agent will access the url associated with the document to find an updated version as well as a review in the language of the user if possible. Furthermore, a second "judge" llm will verify the answer is accurate and no hallucinations occurred when answering the user.


```
trueflag-challenge/
├── pyproject.toml              # Python project configuration and dependencies
├── README.md                   # This file - project overview and guide
├── decisions/                  # Collection of project decisions
├── notebooks/                  # Jupyter notebooks for data exploration
└── src/                        # Proposed solution
    ├──Database/                # Code related to data ingestion and data usage
    |    ├── indexer.py         # Transforms data from CSV into Chroma DB
    |    └── retriever.py       # Class for retrieving documents from Chroma DB
    └── Chatbot/                # Code for agent implementation
    |    ├── agents.py          # Agent definitions (judge and question answering)
    |    ├── nodes.py           # Node and conditional edge definitions
    |    ├── prompts.py         # Collection of prompts used
    |    └── graph.py           # Main entrypoint for running the solution
    ├──config.py                # General configuration
    └──embeddings.py            # Embedding collection
```

# Execution
## Setup

The solution uses uv as the virtual environment manager. To initialize the environment, run:
```bash
uv sync
```

If you are going to run tests and/or notebooks, sync with the dev dependencies.

```bash
uv sync --dev
```

You need a .env file with the GEMINI_API_KEY variable set to a valid API key.

## Run ingestion

To ingest data from CSV into ChromaDB, run the following command:

```bash
uv run --env-file .env python -m src.Database.indexer
```

This process can take several minutes.

## Use Chatbot

The recommended usage of the chatbot is through the user interface:

```bash
uv run --env-file .env python -m src.Chatbot.user_interface
```