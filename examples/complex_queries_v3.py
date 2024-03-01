import json
from typing import List, Dict

from datasets import load_dataset, Dataset

from haystack import Pipeline, Document, component
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
feeding data to another LLM to generate an answer. Examples:

1. Query: Did Microsoft or Google make more money last year?
2. Split: How much profit did Microsoft make? // How much profit did Google make?
3. Get answers for each of the sub-queries and then use them to generate an answer.
       - Sub-query: How much profit did Microsoft make?
       - Answer: Microsoft made 3,141,590 dollars
       - Sub-query: How much profit did Google make?
       - Answer: Google made 2,718,280 dollars
4. Merge results into a structure and feed to LLM to generate the answer: 
    {'query': 'Did Microsoft or Google make more money last year?',
      'sub-questions': [
        {'query': "How much profit did Microsoft make last year?", 'answers': ['3.14159 dollars']},
        {'query': "How much profit did Google make last year?", 'answers': ['2.71828 dollars']}
        ]
    }
5. Ask LLM to generate the answer based on the sub-queries answers.

**Flow**

Query → LLM → decomposed query → run multiple retrievers → merge results → LLM → answer

1. Decompose complex query: the query is split into multiple sub-queries using an LLM.
2. Run multiple retrievers: Each sub-query is run through a retriever to get the top-k documents.
3. Merge results: The results from the retrievers are merged.
4. LLM: The merged results are fed to an LLM to generate an answer.
"""


def get_index_docs(data: Dataset):
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


def build_extractive_retriever_pp(document_store):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()
    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
    return extractive_qa_pipeline


def split_complex_questions_pp():
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

    @component
    class ReplySplitter:

        @component.output_types(queries=List[Document])
        def run(self, replies: List[str]):
            replies = [r.split(' // ') for r in replies]
            return {"queries": [Document(content=subquery) for subquery in replies[0]]}

    builder = PromptBuilder(prompt_template)
    llm = OpenAIGenerator(model="gpt-3.5-turbo")
    splitter = ReplySplitter()
    sentence_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
    sentence_embedder.warm_up()

    pp = Pipeline()
    pp.add_component(name="builder", instance=builder)
    pp.add_component(name="llm", instance=llm)
    pp.add_component(name="splitter", instance=splitter)
    pp.add_component(name="embedder", instance=sentence_embedder)

    pp.connect("builder", "llm")
    pp.connect("llm", "splitter")
    pp.connect("splitter", "embedder")

    return pp


def get_llm_answer_pp():

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

    pp = Pipeline()
    pp.add_component(name="builder", instance=PromptBuilder(prompt_template))
    pp.add_component(name="llm", instance=OpenAIGenerator(model="gpt-3.5-turbo"))
    pp.connect("builder", "llm")

    return pp


class ComplexQueryAnswers:
     
    @component.output_types(queries=Dict)
    def run(self, complex_question: str, queries: List[Document], extractive_retriever_pp: Pipeline):

        # Run the extractive retriever for each of the sub-queries a and fill in the collected_answers dict
        collected_answers = {
            "query": complex_question,
            "subqueries": []
        }

        for doc in queries:
            results = extractive_retriever_pp.run(
                data={"retriever": {'query_embedding': doc.embedding, 'top_k': 3},
                      "reader": {"query": doc.content, "top_k": 2}}
            )
            if results:
                answers = {"query": doc.content, "answers": []}
                for answer in results['reader']['answers']:
                    if answer.data:
                        answers['answers'].append(answer.data)
                collected_answers['subqueries'].append(answers)

        return collected_answers


def main():
    # Load data and index it in the document store, and create an extractive retriever
    data = load_dataset("Tuana/game-of-thrones")
    doc_store = get_index_docs(data)
    extractive_retriever_pp = build_extractive_retriever_pp(doc_store)

    # query = "Who has more sibilings, Jaime Lannister or Jonh Snow?"
    query = "Which family has more members, the Lannisters or the Starks?"

    pp_split = split_complex_questions_pp()
    pp_split.run(data={"question": query})
    results = pp_split.run(data={"question": query})
    queries = results['embedder']['documents']

    pp = Pipeline()
    complex_query = ComplexQueryAnswers()
    pp.add_component(name="complex_query", instance=complex_query)


    """
    # Run the extractive retriever for each of the sub-queries a and fill in the collected_answers dict
    collected_answers = {
        "query": query,
        "subqueries": []
    }

    for doc in queries:
        results = extractive_retriever_pp.run(
            data={"retriever": {'query_embedding': doc.embedding, 'top_k': 3},
                  "reader": {"query": doc.content, "top_k": 2}}
        )
        if results:
            answers = {"query": doc.content, "answers": []}
            for answer in results['reader']['answers']:
                if answer.data:
                    answers['answers'].append(answer.data)
            collected_answers['subqueries'].append(answers)
    """

    # Run the LLM to generate the answer for the complex query based on the sub-queries answers
    answer_pp = get_llm_answer_pp()
    answer_pp.run(data={"question": json.dumps(collected_answers)})


if __name__ == '__main__':
    main()

