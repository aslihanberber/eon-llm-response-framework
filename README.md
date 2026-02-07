# eon-llm-response-framework
Aslihan Berber
Mohammed Yazan Aljleilati
TUM WS 25/26

# E.ON Customer Email Drafting Assistant  
Human-in-the-loop LLM system for topic-aware, tone-controlled customer service email drafting

## Overview

This repository contains the full implementation and project report for a customer service email drafting assistant developed in the context of an E.ON use case. The project investigates whether a compact, open-source large language model can be adapted to support customer service agents under realistic enterprise constraints, including limited data availability, constrained computational resources, and strict safety requirements.

The implemented system combines topic modeling, lightweight rule-based summarization, and parameter-efficient fine-tuning of a compact language model to generate structured, tone-controlled draft responses. All generated outputs are intended to support, not replace, human customer service agents.

The project is implemented entirely in Python and executed in a Google Colab (pay-as-you-go) environment using an NVIDIA L4 GPU.


## Repository Structure

├── group10_final_code_eon_llm_powered_email_drafting.ipynb
│   └── Main Google Colab notebook (recommended entry point).
│       Contains the full pipeline including data loading, BERTopic training,
│       synthetic data generation, model fine-tuning (commented by default),
│       evaluation, and Streamlit deployment.

├── group10_final_code_eon_llm_powered_email_drafting.py
│   └── Python script version of the notebook, exported from Google Colab.
│       Useful for code inspection or local adaptation.

├── model-001.safetensors
│   └── (Not included in the public repository due to NDA restrictions.)
│       Saved merged fine-tuned TinyLlama model weights (QLoRA adapters merged).
│       Used during development for inference without retraining.

├── tinyllama_eon_merged.zip
│   └── (Not included in the public repository due to NDA restrictions.)
│       Compressed archive of the merged fine-tuned TinyLlama model directory.

├── GoldenDatasetSyntheticAnswers.zip
│   └── (Not included in the public repository due to NDA restrictions.)
│       Synthetic instruction–response dataset generated from anonymized
│       customer emails using external LLMs for instruction fine-tuning.
│       The generation process and data format are fully described in the report.

├── Group10_Report_Enhancing Customer Service Through LLM-Powered Email Drafting An E.ON Data and Analytics Use Case.pdf
│   └── Final written project report.

├── Group10_Final Presentation Enhancing Customer Service Through LLM-Powered Email Drafting An E.ON Data and Analytics Use Case.pdf
│   └── Final project presentation (PDF version).

├── Group10_Final Presentation Enhancing Customer Service Through LLM-Powered Email Drafting An E.ON Data and Analytics Use Case.pptx
│   └── Final project presentation (PowerPoint version).

├── final_app_recording_for_final_presentation.wmv
│   └── Screen recording demonstrating the Streamlit web application
│       running inside Google Colab.

└── README.md
    └── Project documentation and usage instructions.



## Recommended Execution Environment

This project is designed to be run in **Google Colab**.

- Platform: Google Colab (pay-as-you-go)
- GPU: NVIDIA L4
- Python version: Python 3
- Key libraries:
  - transformers, peft, trl, bitsandbytes
  - sentence-transformers
  - BERTopic, HDBSCAN, UMAP
  - PyTorch
  - Streamlit

For best results, open `group10_final_code_eon_llm_powered_email_drafting.ipynb` directly in Google Colab and enable a GPU runtime.


## End-to-End Pipeline

The system follows the pipeline below:

1. **Customer Email Ingestion**
   - Anonymized German customer emails are loaded from text files.
   - Emails are converted into a structured JSON Lines format.

2. **Topic Modeling**
   - BERTopic is trained on the customer email corpus.
   - Multilingual sentence embeddings are used for clustering.
   - Custom German and domain-specific stopwords are applied.

   **Note:**  
   The BERTopic training step is not commented out. As a result, topic distributions and topic IDs may vary slightly between runs.

3. **Synthetic Instruction Dataset Construction**
   - The dataset contains only incoming customer emails (no agent replies).
   - Synthetic responses are generated using multiple external LLMs.
   - Each email receives:
     - One formal response
     - One friendly response
   - All synthetic responses are manually reviewed and validated.

4. **Model Fine-Tuning (QLoRA)**
   - Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - Parameter-efficient fine-tuning using LoRA adapters
   - 4-bit quantization (QLoRA)
   - Fine-tuning is performed under strict hardware constraints.

   **Important:**  
   The fine-tuning and training cells are commented out by default.  
   The notebook loads a previously saved, merged fine-tuned model to avoid retraining on every execution.

   To train the model from scratch, simply uncomment the fine-tuning sections in the notebook.

5. **Inference Pipeline**
   For each new customer email:
   - Topic assignment using the trained BERTopic model
   - Short rule-based summary (greeting and signature removal)
   - Response drafting using the fine-tuned TinyLlama model
     - Formal version
     - Friendly version
   - Deterministic decoding with explicit safety and compliance rules

6. **Evaluation and Validation**
   - Lexical diversity (entropy)
   - Semantic similarity (Sentence-BERT)
   - Rule-based hallucination checks for:
     - Numbers
     - Dates
     - Email addresses
     - URLs
     - Phone numbers

7. **Streamlit Web Application**
   - A Streamlit-based UI is launched inside Google Colab.
   - The app appears in the output of the final notebook cell.
   - No external hosting or deployment is required.

   The application allows customer service agents to:
   - Paste customer emails
   - Select response tone (formal or friendly)
   - View topic labels and summaries
   - Edit generated drafts
   - Copy finalized responses


## Reproducibility Notes

- Topic modeling results may vary slightly across runs due to retraining of BERTopic.
- Core model behavior, safety constraints, and inference logic remain stable.
- All generated responses are intended for mandatory human review.

This repository is intended for academic evaluation and demonstration purposes and does not represent a production-ready system.


## Intended Use

The system is explicitly designed as:
- A drafting assistant
- A decision-support tool
- A human-in-the-loop system

It is not intended for fully automated customer communication.

## Context

This project was developed as part of an academic course project using an E.ON customer service use case and anonymized customer data. The repository includes both the full technical implementation and the final written report submitted for evaluation.

## Data Availability and NDA Notice

This project was conducted under a non-disclosure agreement (NDA) with E.ON. As a result, raw customer emails and derived training datasets are not included
in this repository. All data used in the project was anonymized, and the full data generation and modeling methodology is documented in the accompanying report. The fine-tuned model weights are not distributed publicly as well. Users may retrain the model by uncommenting the training and fine-tuning sections in the provided notebooks. The application demo video uses synthetic example input only. No real customer emails or proprietary E.ON data are shown.


