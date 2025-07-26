from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from transformers import pipeline
import torch

# Memory
memory = ConversationBufferMemory(return_messages=True)

# HF Pipelines
query_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
segmenter = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text_generator = pipeline( "text2text-generation", model="google/flan-t5-base")

# Define LangGraph State
class GraphState(TypedDict):
    query: str
    context: str
    segment: str
    response: str

# Node 1: Query Handler + Context
def handle_query(state: GraphState) -> GraphState:
    query = state['query']
    memory.chat_memory.add_user_message(query)
    context = "\n".join([m.content for m in memory.chat_memory.messages])
    return {**state, "context": context}

# Node 2: Segment Customer using Zero-Shot Classification
def segment_customer(state: GraphState) -> GraphState:
    query = state["query"]
    candidate_labels = ["Price-Sensitive", "Premium", "Value-Oriented"]
    result = segmenter(query, candidate_labels)
    segment = result["labels"][0]  # most likely class
    return {**state, "segment": segment}

# Node 3: Generate Suggestion using HF Text Generator

def suggest_response(state: GraphState) -> GraphState:
    segment = state['segment']
    prompt = (
        f"The customer is categorized as a '{segment}' customer. "
        "Suggest a product plan that best fits their needs and briefly explain your reasoning."
    )
    response = text_generator(
        prompt,
        max_length=100,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']
    
    return {**state, "response": response.strip()}


# Build LangGraph
builder = StateGraph(GraphState)

builder.add_node("query_handler", handle_query)
builder.add_node("segmenter", segment_customer)
builder.add_node("responder", suggest_response)

builder.set_entry_point("query_handler")
builder.add_edge("query_handler", "segmenter")
builder.add_edge("segmenter", "responder")
builder.add_edge("responder", END)

graph = builder.compile()
output = graph.invoke({"query": "I'm looking for a high-quality premium plan with features."})
print("Segment:", output["segment"])
print("Response:", output["response"])
