# Smart MCQ Solver: Deep Learning & Generative AI Project

**Author:** Tanmay Hatkar  
**Roll Number:** 24f2008463  
**Course:** BSDA2001P - Diploma Level of BS in Data Science and Applications  

---

## 📖 Project Overview
This repository contains the end-to-end machine learning pipeline for the Smart MCQ Solver Challenge. The objective of this project is to build an intelligent system capable of understanding complex context, reasoning across multiple options, and accurately ranking the top three most probable answers for challenging multiple-choice questions. 

The pipeline evolves from traditional NLP statistical methods to state-of-the-art Transformer architectures, incorporating Retrieval-Augmented Generation (RAG) and Low-Rank Adaptation (LoRA) fine-tuning.

---

## 🚀 Milestone Tracking
*This section tracks the progression of the project requirements.*

- [x] **Milestone 1:** NLP Foundations & Semantic Similarity (TF-IDF, Cosine Similarity)
- [x] **Milestone 2:** Transformers & Dense Context-Aware Embeddings (MiniLM, Zero-Shot Classification)
- [x] **Milestone 3:** Context Augmentation with RAG Pipelines (FAISS, Bi-Encoders, Cross-Encoder Reranking)
- [ ] **Milestone 4:** Formulating MCQ Tasks & LoRA Fine-Tuning 
- [ ] **Milestone 5:** Ensembling & Final Deployment

---

## 📂 Repository Structure
```text
project-name/
│
├── notebooks/
│   ├── milestone-1.ipynb
│   ├── milestone-2.ipynb
│   ├── milestone-3.ipynb
│   └── final_notebook.ipynb
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── reports/
│   ├── milestone-1-report.pdf
│   ├── milestone-2-report.pdf
│   └── final-report.pdf
│
├── models/
│
├── requirements.txt
└── README.md