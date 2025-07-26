# LangGraph: Customer Segmentation and Recommendation with LangChain + HuggingFace

## Overview
A minimal but powerful pipeline built using LangGraph, LangChain Memory, and Hugging Face Transformers to:

Track customer queries

Segment them (e.g., Premium, Price-Sensitive, Value-Oriented)

Generate smart, personalized product plan suggestions

## Features
- [x] Track customer queries
- [x] Segment them (e.g., Premium, Price-Sensitive, Value-Oriented)
- [x] Generate smart, personalized product plan suggestions
- [x] Easy to use and customize
- [x] Works with LangChain Memory
- [x] Works with Hugging Face Transformers
- [x] Works with LangGraph

## Setup
1. Clone the repository:
```bash
git clone https://github.com/Arbin17/Langgraph.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Run the pipeline:
```bash
python main.py
```
## LangGraph Flow (MermaidJS)
graph TD;
-  __start__ --> capture_query;
-  capture_query --> segment_customer;
-  segment_customer --> suggestion;
-  suggestion --> __end__;

## License
This project is licensed under the MIT License.
