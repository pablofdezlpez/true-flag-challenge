# Reduce the amount of metadata in dataset

## Status

Proposed

## Context

An exploration of the data has been done as seen in this [notebook](../notebooks/01-initial-dataset-exploration.ipynb)

## Decision

The solution is going to limit the information to document title, text, summary, url and images.

## Consequences

We are discarting possible relevant information in video as well as being able to reference the authors directly. However, video comprehension would require a much detailed solutions, and authority can be taken from the url.
