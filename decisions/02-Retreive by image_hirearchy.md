# Retreive by image / hirearchy

## Status

Proposed

## Context

It is expected that the user will use either a text query or image query for detecting "fake" news. We have at our disposal images, summary and texts and need to decide how are we retrieving the documents

## Decision

If user inputs an image, the retrieval process will look for images. If none is found, it would default to summary. This is done because the expectation is that the user is researching a "known" fake new that is being shared in image format.

When querying with text (or if image is not found) the system will first query by summary and then chunks of texts from the selected summaries. Summaries expose whether the news is fake or not, while the text will give a detail explanation.

## Consequences

This querying process might be slow if images are not relevant, since it will pass three times. But it is expected to yield more accurate documentation
