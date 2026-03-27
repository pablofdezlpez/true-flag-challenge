# Agentic RAG (for Claim Reviews)

## Context

A ClaimReview is a metadata standard to capture relevant information about a fact-check
(e.g. title, image, rating, claim…). Because the claim review document combines text and
multimedia resources, a search engine or assistance chatbot should take into account not
only the textual information, but also all the information that can be extracted from
multimedia resources.

In this context, the objective of the challenge is to build an Agentic RAG
system that, when faced with a question asked in text or
image:

1. Retrieves the most relevant claim review candidates
2. Generates a reasoned response.

In both phases, you should consider: (1) the use of a multi-agent architecture, (2) the use of images, and (3) expanded
context information from external sources.

__NOTE__: If you need more information about ClaimReview schema, please
access [Fact Check (ClaimReview) Markup for Search](https://developers.google.com/search/docs/appearance/structured-data/factcheck?hl=es).

## Task definition

The objective is to develop an Agentic RAG (Retrieval-Augmented Generation) system able to generate a
response using textual information from the provided ClaimReview documents[^1] and other contextual information from the dataset or external sources. Please, refer [The Data](#the-data).

A fact-checking or misinformation verification process must be reliable, provide verifiable evidence, and reach a conclusion about its accuracy and precision. It is especially important that the system output provides accurate information in the generated response. Therefore, **the core value of the proposed solution should focus on the detection or mitigation of hallucinations and on robust evaluation** within a RAG framework. Proposals that contribute to improving model reliability, factual consistency, or evaluation methodology in these areas will be particularly valued.

Please include a document (Markdown or PDF) explaining your reasoning process, assumptions, and technical decisions about the implemented proposal. Other proposals, approaches or ideas not implement could be discussed later.

[^1]: The provided ClaimReview documents are an extract of our claim review database with records including claim
review, derived textual data as summary or keywords, and image that contextualize or complement the
article (see below the dataset description).


### Other considerations

- The system should always respond that it does not have enough information to respond to the
  user if it does not find relevant documents in the search space (i.e. the dataset provided).
  External information should only be used to expand the context.
- We are looking for "production-oriented" code (no notebooks), in the sense of providing a minimal structure, OOP, a
  dependency / virtual environment manager, and at least one entrypoint for launching new textual queries
  (or batched through a CSV file).


## The Data

The attached dataset contains verified/reviewed claims available in our claim review database.

It includes contextual information about these verifications/reviews, such as full text, summary,
url, description, keywords, dates or attached multimedia resources (image, video).

Also, each
record includes a similarity relation (labelled by humans) indicating whether the unverified claim matches (“similarity” = 1) or doesn't match (“similarity” = 0) the claim review.

### Field description

#### Basic info

* **reviewed claim** *(str, multilingual)*: reviewed claim
* **title** *(str, multilingual)*: article title
* **text** *(str, multilingual)*: original text. It can contains "noise" in the sense of including
  some incorrect paragraphs (texts of menus, whitespaces, symbols, etc.)
* **summary** *(str, multilingual)*: summary generated from original text (using an external service)

#### Multimedia resources

* **cr_image** *(str, url)*: (if any) image attached into the claim review
* **meta_image** *(list[str], url)*: (if any) linked images extracted from the meta and/or json+ld object
* **movies** *(list[str], url)*: (if any) videos linked on the meta and/or json+ld object

#### Other metadata

* **meta_description** *(str, multilingual)*: description extracted from meta tags and/or json+ld representation
* **kb_keywords** *(list[str], multilingual)*: keywords generated from original text (ngrams range 1-3) using KeyBert
* **meta_keywords** *(list[str], multilingual)*: keywords extracted from meta tags and/or json+ld representation
* **url** *(string, url)*: url of the claim review
* **domain** *(string, url)*: domain (of the publisher)
* **published** *(datetime)*: publication date
* **cm_authors** *(list[str], multilingual)*: a list of authors (extracted from meta tags and/or json+ld representation)
* **cr_author_name** *(str): name of the author of the claim review (the fact-checker)
* **cr_country** *(string)*: country of publication
* **meta_lang** *(string, ISO Code)* native language of the claim review extracted from meta tags

#### Similarity relation

* **unverified claim** *(str, multilingual (en, es))*: unverified claim / search term
* **similarity** *(int)*: label indicating a positive (1) or negative (0) match between a search term
  (or unverified claim) and the verified claim


## Important Notice on the Use of AI Tools
The use of support or code‑generation tools powered by artificial intelligence (such as ChatGPT, GitHub Copilot, or similar) is not restricted during this technical test. We understand these tools are part of many developers’ regular workflows. However, the final solution must be original and reflect your own technical judgment, design skills, and problem‑solving ability.

Submissions that are fully or predominantly generated by AI tools will be penalized, as will those that show no clear evidence of understanding or reasoning behind technical decisions. Likewise, the inclusion of files, code fragments, or resources that are out of context or irrelevant to the objectives of the test will also be penalized, particularly when their purpose within the presented solution cannot be properly explained.
