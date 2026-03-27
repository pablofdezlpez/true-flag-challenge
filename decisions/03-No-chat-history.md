# Chatbot will be of a single pass (no chat history)

## Status

Proposed

## Context

Usually in chatbot implementations, the user would be able to have a back and forth with the chatbot. However, in this application the user is querying whether a news article is false

## Decision

Chatbot will have one pass in attempting to answer the user. In practice, the system queries a database of fake news and returns a found article that disproves the news (or not)

## Consequences

User will not be able to inquiry further on the topic. Implementation is also faster.
