import json
from typing import List

from datasets import load_dataset, Dataset

from haystack import Pipeline, Document
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

embedder_model = "sentence-transformers/all-MiniLM-L6-v2"

"""
**Description**

This pipeline should split complex queries into subqueries and then run retrieval for each of the sub-queries before 
feeding data to another LLM to generate an answer.

**Flow**

Query → LLM → decomposed query → run multiple retrievers → merge results → LLM → answer

1. decomposed query: The query is split into multiple sub-queries.
2. run multiple retrievers: Each sub-query is run through a retriever to get the top-k documents.
3. merge results: The results from the retrievers are merged.
4. LLM: The merged results are fed to an LLM to generate an answer.
"""


@component
class ReplySplitter:

    @component.output_types(queries=List[str])
    def run(self, replies: List[str]):
        return {"queries": [r.split(' // ') for r in replies]}


def index_docs(data: Dataset):
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()

    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedder_model))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy="skip"))

    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run({"cleaner": {"documents": [Document.from_dict(doc) for doc in data["train"]]}})

    return document_store


def query_retriever(doc_store: InMemoryDocumentStore, top_k: int = 3, ):

    sentence_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store, query_embedding_model=embedder_model,
                                           passage_embedding_model=embedder_model, top_k=top_k)
    pipeline = Pipeline()
    pipeline.add_component("embedder", sentence_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("embedder", "retriever")

    return pipeline


def extractive_retriever(document_store):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()
    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
    return extractive_qa_pipeline


def build_pipeline():
    prompt_template = """
    You are a query engine.
    You prepare queries that will be send to a web search component.
    Sometimes, these queries are very complex.
    You split up complex queries into multiple queries so that you can run multiple searches to find an answer.
    When you split a query, you separate the sub-queries with '//'.
    If the query is simple, then keep it as it is.
    ###
    Example 1:
    Query: Did Microsoft or Google make more money last year?
    Split: How much profit did Microsoft make? // How much profit did Google make?
    ###
    Example 2:
    Query: What is the capital of Germany?
    Split: What is the capital of Germany?
    ###
    Example 3:
    Query: {{question}}
    Split:
    """

    builder = PromptBuilder(prompt_template)
    llm = OpenAIGenerator(model="gpt-3.5-turbo")
    splitter = ReplySplitter()

    pp = Pipeline()
    pp.add_component(name="builder", instance=builder)
    pp.add_component(name="llm", instance=llm)
    pp.add_component(name="splitter", instance=splitter)
    # pp.add_component(name="multiplexer", instance=Multiplexer(str))

    pp.connect("builder", "llm")
    pp.connect("llm", "splitter")
    # pp.connect("splitter", "multiplexer.value")

    return pp


def get_llm_answer():
    prompt_template = """
    You answer a complex query which was split into multiple sub-questions.
    You inspect the sub-questions answers to generate an answer for the complex question.
    The sub-questions can have multiple answers and you need to merge them or select the best one or discard some.
    The query and the sub-questions are provided as JSON data.
    ###
    Example 1:
    Complex Question: "{'query': 'Did Microsoft or Google make more money last year?',
                       'sub-questions': [
                            {'query': "How much profit did Microsoft make last year?", 'answers': ['3.14159 dollars']},
                            {'query': "How much profit did Google make last year?", 'answers': ['2.71828 dollars']}
                        ]
                       }"
    Answer: Microsoft made more money last year.
    ###
    Example 2:
    Example 1:
    Complex Question: "{'query': 'Who's older Joe Biden or Donald Trump?',
                       'sub-questions': [
                                {'query': "How old is Joe Biden?", 'answers': ['81 years old']},
                                {'query': "How old is Donald Trump?", 'answers': ['77 years old']}
                        ]
                       }"
    Answer: Joe Biden is older than Donald Trump.
    ###
    Example 3:
    Complex Question: {{question}}
    Answer:
    """

    builder = PromptBuilder(prompt_template)
    llm = OpenAIGenerator(model="gpt-3.5-turbo")

    pp = Pipeline()
    pp.add_component(name="builder", instance=builder)
    pp.add_component(name="llm", instance=llm)

    pp.connect("builder", "llm")

    return pp


def main():
    # query = "Who has more sibilings, Jaime Lannister or Jonh Snow?"
    query = "Which family has more members, the Lannisters or the Starks?"

    pp_split = build_pipeline()
    pp_split.run(data={"question": query})
    results = pp_split.run(data={"question": query})
    queries = results['splitter']['queries'][0]

    data = load_dataset("Tuana/game-of-thrones")
    doc_store = index_docs(data)

    sentence_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
    sentence_embedder.warm_up()
    embedded_queries = sentence_embedder.run([Document(content=q) for q in queries])
    extractive_retriever_pp = extractive_retriever(doc_store)

    answer_pp = get_llm_answer()

    collected_answers = {
        "query": query,
        "subqueries": []
    }

    for q in embedded_queries['documents']:
        results = extractive_retriever_pp.run(data={
            "retriever": {'query_embedding': q.embedding, 'top_k': 3},
            "reader": {"query": q.content, "top_k": 2}})

        answers = {"query": q.content, "answers": []}
        for answer in results['reader']['answers']:
            if answer.data:
                answers['answers'].append(answer.data)
        collected_answers['subqueries'].append(answers)

    answer_pp.run(data={"question": json.dumps(collected_answers)})


if __name__ == '__main__':
    main()

