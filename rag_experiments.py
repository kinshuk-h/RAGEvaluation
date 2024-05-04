import os
import re
import sys
import math
import string
import logging
import itertools
import functools

import torch
import datasets
import sqlalchemy
import llama_index
import nest_asyncio
import nltk.tokenize
import tqdm.auto as tqdm

import llama_index.core
import llama_index.core.ingestion
import llama_index.core.node_parser

from llama_index.core import (
    Settings, VectorStoreIndex, StorageContext
)

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.schema import TransformComponent, QueryBundle
from llama_index.core.evaluation.benchmarks.hotpotqa import f1_score, exact_match_score

from evaluation.models import ModelInference, HuggingFaceModelInference, OpenAIModelInference

# =====

FORMATTER = string.Formatter()

# INITIAL_QUERY_PROMPT_ARGS = dict(
#     instructions=[
#         "Given a query as input, answer it using the additional set of context documents provided alongside.",
#         "Only answer the question if it is possible to do so with the provided context.",
#         "Respond with 'I don't know' if the context does not have enough information to answer the question.",
#         "Give a brief response, with as few words as possible. If the question asks to compare, the answer can be a simple yes or no."
#     ],
# )
# INITIAL_QUERY_PROMPT_FORMAT = "Context:\n\n{context}\n\nQuery: {query} Give a short factoid answer (as few words as possible)"
# REFINE_QUERY_PROMPT_ARGS = dict(
#     instructions=[
#         "Given a query and an initial draft for an answer, refine it based on information inferred from the context documents provided alongside.",
#         "If the answer is already sufficient, then do nothing and give the answer as-is. Otherwise improve it using the new information.",
#         "Give a brief response, with as few words as possible. If the question asks to compare, the answer can be a simple yes or no."
#     ]
# )
# REFINE_QUERY_PROMPT_FORMAT = "New Context:\n\n{context}\n\nQuery: {query} Give a short factoid answer (as few words as possible)\nAnswer:{answer}"

INITIAL_QUERY_PROMPT_ARGS = dict(
    instructions=[],
    prompt_template="{instructions}{examples}{input}"
)
INITIAL_QUERY_PROMPT_FORMAT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query} Give a short factoid answer (as few words as possible).\n"
    "Answer: "
)
REFINE_QUERY_PROMPT_ARGS = dict(
    instructions=[],
    prompt_template="{instructions}{examples}{input}",
)
REFINE_QUERY_PROMPT_FORMAT = (
    "The original query is as follows: {query} Give a short factoid answer (as few words as possible).\n"
    "We have provided an existing answer: {answer}\n"
    "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "------------\n"
    "{context}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"</?s>", "", node.text)
        return nodes

def make_id_from_title(title):
    return re.sub(r"(?u)\s+", "-", title.lower())

def make_id(index, document):
    return document.doc_id + "-" + str(index)

def is_number(x):
    try: int(x); return True
    except: return False

def format_prompt(prompt_template, *args, **kwargs):
    if isinstance(prompt_template, list):
        new_prompt_template = []
        for turn in prompt_template:
            new_turn = { **turn }
            new_turn['content'] = format_prompt(new_turn['content'], *args, **kwargs)
            new_prompt_template.append(new_turn)
        prompt_template = new_prompt_template
    else:
        parsed_fields =[ item for _, item, _, _ in FORMATTER.parse(prompt_template) if item ]

        repl_dict = { item: f"{{{item}}}" for item in parsed_fields if not is_number(item) }
        repl_dict.update(**{ key: value for key, value in kwargs.items() if key in repl_dict })

        repl_args = [ f"{{{item}}}" for item in parsed_fields if is_number(item) ]
        repl_args[:len(args)] = args

        prompt_template = prompt_template.format(*repl_args, **repl_dict)
    return prompt_template

# engine = sqlalchemy.create_engine("sqlite:////home/kinshuk/projects/e2-355-tai-rag/data/enwiki-20230401.db")
# raw_documents = None

# def get_documents(limit=None):
#     global raw_documents

#     with engine.connect() as connection:
#         row_count = connection.execute(sqlalchemy.text("SELECT COUNT(title) FROM documents")).fetchone()[0]
#         if raw_documents is None:
#             raw_documents = []
#             for row in tqdm.tqdm(connection.execute(sqlalchemy.text("SELECT * FROM documents")), total=row_count):
#                 text = row[1].replace("####SPECIAL####SEPARATOR####", "")
#                 raw_documents.append(llama_index.core.Document(doc_id=make_id_from_title(row[0]), text=text, extra_info={ 'title': row[0] }))
#         return raw_documents

def get_index(embedding_model, documents, split_type, embedding_type, split_size=None) -> VectorStoreIndex:

    index_path = f"data/indexes/{split_type}-{embedding_type}{'-' + str(split_size) if split_size is not None else ''}"

    if os.path.exists(index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = llama_index.core.load_index_from_storage(storage_context)

    else:
        if split_type == "semantic":
            split_transform = llama_index.core.node_parser.SemanticSplitterNodeParser(
                embed_model=embedding_model, id_func=make_id
            )
        else:
            split_transform = llama_index.core.node_parser.SentenceSplitter(
                chunk_size=split_size, chunk_overlap=0, id_func=make_id
            )
        pipeline = llama_index.core.ingestion.IngestionPipeline(
            transformations=[ split_transform, TextCleaner(), embedding_model ]
        )

        results = pipeline.run(show_progress=True, documents=documents, num_workers=1)

        index = llama_index.core.VectorStoreIndex(results)
        index.storage_context.persist(index_path)

    return index

def make_llama_index_llm(model: ModelInference):
    if isinstance(model, HuggingFaceModelInference):
        model.load()
        return HuggingFaceLLM(model=model.model, tokenizer=model.tokenizer)
    else:
        return AzureOpenAI(model.name, **{
            'azure_endpoint'  : os.environ['AZURE_OPENAI_ENDPOINT'],
            'api_key'         : os.environ['AZURE_OPENAI_API_KEY'],
            'azure_deployment': os.environ['AZURE_OPENAI_DEPLOYMENT'],
            'api_version'     : os.environ['AZURE_OPENAI_API_VERSION']
        })

def make_retriever(mode, index: VectorStoreIndex, k, embed_model):
    return index.as_retriever(embed_model=embed_model, similarity_top_k=k, show_progress=False)

def make_query_engine(mode, index: VectorStoreIndex, k, llm, embed_model):
    return index.as_query_engine(llm=llm, embed_model=embed_model, similarity_top_k=k, show_progress=False)

def get_dataset(*dataset):
    if dataset[0] == "hotpot_qa":
        return datasets.load_dataset(*dataset)['validation']
    elif dataset[0] == "custom":
        return [
            {
                'question': 'Who is Jeff Hammond?',
                'answer': 'Nobody knows',
                'id': 'only-one',
                'category': 'only-one'
            }
        ]

def normalize_instance(dataset_name, instance):
    if dataset_name == "hotpot_qa":
        return {
            'id'      : f"{instance['id']}",
            # 'category': f"{instance['level']}",
            'question': instance['question'],
            'answer'  : instance['answer']
        }
    return instance

def format_node(node):
    if len(node.get('meta', {})):
        metadata_desc = '\n'.join(f"{key}: {value}" for key, value in node['meta'].items())
        return f"{metadata_desc}\n{node['text']}"
    else:
        return node['text']

def compactify_context(model, prompt, retrieved_context, max_context_length=2048):
    new_retrieved_context = []
    for node in retrieved_context:
        sentences = nltk.tokenize.sent_tokenize(node['text'])
        sent_count = len(sentences)
        if len(node.get('meta', {})):
            metadata_desc = '\n'.join(f"{key}: {value}" for key, value in node['meta'].items()) + '\n'
        else:
            metadata_desc = ''
        offset = 0
        while len(sentences) > 0:
            index = len(sentences)
            while index > 1:
                tokens = model.tokenize(format_prompt(prompt, context=metadata_desc + ' '.join(sentences[:index])))
                if len(tokens) <= max_context_length: break
                index -= 1
            node_id = node['id']
            if offset != 0 or offset + index < sent_count: node_id += f"-{offset}:{offset+index}"
            new_retrieved_context.append({
                'id'  : node_id,
                'meta': node['meta'],
                'text': ' '.join(sentences[:index]),
            })
            offset += index
            sentences = sentences[index:]
    return new_retrieved_context

def get_refine_turns(model, initial_prompt, refine_prompt, retrieved_context, max_context_length=2048):
    refine_turns = []
    offset, prompt = 0, initial_prompt
    while len(retrieved_context) > 0:
        index = len(retrieved_context)
        while index > 1:
            tokens = model.tokenize(format_prompt(prompt, context='\n\n'.join(retrieved_context[:index])))
            if len(tokens) <= max_context_length: break
            index -= 1
        refine_turns.append((offset, offset+index))
        offset += index
        prompt = refine_prompt
        retrieved_context = retrieved_context[index:]
    return refine_turns

def answer_with_rag(model, queries, retrieved_contexts, max_context_length=2048, **generation_kwargs):
    initial_prompts = [
        model.make_prompt(
            format_prompt(INITIAL_QUERY_PROMPT_FORMAT, query=query),
            **INITIAL_QUERY_PROMPT_ARGS
        )[0]
        for query in queries
    ]
    refine_prompts = [
        model.make_prompt(
            format_prompt(REFINE_QUERY_PROMPT_FORMAT, query=query),
            **REFINE_QUERY_PROMPT_ARGS
        )[0]
        for query in queries
    ]

    retrieved_contexts = [
        compactify_context(model, prompt, context, max_context_length)
        for prompt, context in zip(initial_prompts, retrieved_contexts)
    ]
    retrieved_contexts = [
        [ format_node(node) for node in context ]
        for context in retrieved_contexts
    ]

    refine_turns = [
        get_refine_turns(model, initial_prompt, refine_prompt, context, max_context_length)
        for initial_prompt, refine_prompt, context in zip(initial_prompts, refine_prompts, retrieved_contexts)
    ]

    answers = model.generate([
        format_prompt(prompt, context='\n\n'.join(context[turn[0][0]:turn[0][1]]))
        for prompt, context, turn in zip(initial_prompts, retrieved_contexts, refine_turns)
    ], **generation_kwargs)

    level = 1
    while True:
        index = [ i for i in range(len(refine_turns)) if len(refine_turns[i]) > level ]

        if len(index) == 0: break

        prompts_filt  = [ refine_prompts[i] for i in index ]
        turns_filt    = [ refine_turns[i] for i in index ]
        contexts_filt = [ retrieved_contexts[i] for i in index ]
        answers_filt  = [ answers[i] for i in index ]

        new_answers = model.generate([
            format_prompt(prompt, answer=answer, context='\n\n'.join(context[turn[level][0]:turn[level][1]]))
            for prompt, answer, context, turn in zip(prompts_filt, answers_filt, contexts_filt, turns_filt)
        ], **generation_kwargs)

        for idx, answer in zip(index, new_answers):
            answers[idx] = answer

        level += 1

    return answers