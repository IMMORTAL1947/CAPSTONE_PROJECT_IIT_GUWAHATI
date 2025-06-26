# CAPSTONE_PROJECT_IIT_GUWAHATI
:  🧠 AI Customer Support Assistant A fine-tuned, lightweight AI chatbot built to automate customer service queries using generative models (Mistral 7B) and LoRA fine-tuning.
🚀 Project Overview
This project demonstrates an end-to-end pipeline to build an AI-powered customer support system:

Fine-tunes a large language model on real-world customer queries .

Implements LoRA (Low-Rank Adaptation) for memory-efficient training .

Deploys the model with a lightweight Streamlit web interface for real-time customer interaction.

💡 Key Features
✨ Fine-tuned Mistral 7B on 27k+ cleaned customer service queries

💬 Understands and responds to common support questions (orders, returns, policies, etc.)

🧠 Uses Hugging Face Transformers + PEFT + BitsAndBytes (4-bit quantized)

🔁 Easily deployable via Streamlit frontend

🧱 Efficient training via Google Colab + LoRA

📦 Tech Stack
Model: mistralai/Mistral-7B-Instruct-v0.1 + LoRA

Libraries: Hugging Face Transformers, PEFT, Streamlit, BitsAndBytes

Training: Google Colab with GPU acceleration

Frontend: Streamlit UI with real-time query interface
