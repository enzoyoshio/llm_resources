# LLM Research Themes for a 9‑Month Lab Project

## 🧪 Theme 1: Parameter-Efficient Fine‑Tuning for a Specific Domain

**Why it fits you:** Directly uses the low‑rank adaptation intuition of LoRA — conceptually simple yet powerful. You can implement, benchmark, and systematically vary configurations without expensive hardware. LoRA reduces trainable parameters to **1–5%** of the original model.

| Aspect | Details |
|--------|---------|
| Core research question | How do LoRA hyperparameters (rank r, scaling factor α, target modules, learning rate) affect performance vs. computational cost for a specific domain? |
| Domain choices | Fine‑tune a small base model (e.g., Llama 3.2 3B, Phi‑3 mini 3.8B, Qwen2.5‑1.5B) on a moderately sized task (legal/medical document classification, scientific abstract summarisation, or sentiment analysis on a non‑English language). |
| Novice‑friendly aspect | LoRA is well‑documented, there are many open‑source Colab notebooks and GitHub repositories that show end‑to‑end pipelines. You can start with classification (SST‑2, IMDB) and move to generation (summarisation on XSum). |
| 9‑month roadmap | **Months 1–2:** Literature review on PEFT (LoRA, QLoRA, AdaLoRA). **Months 3–4:** Implement a baseline with full fine‑tuning on a small model, then a LoRA implementation using Hugging Face PEFT. **Months 5–7:** Perform a hyperparameter sweep (rank r = 4,8,16,32; different target modules). Measure memory usage, training time, inference speed, and downstream metric (accuracy, ROUGE). **Month 8:** Write up results and create visualisations (e.g., trade‑off curves). **Month 9:** Prepare final report. |
| Potential extensions | Compare QLoRA (quantised LoRA) – runs on a single consumer GPU with 6–12 GB VRAM. Evaluate on a **low‑resource language** (e.g., Azerbaijani sentiment analysis) where full fine‑tuning overfits. |

---

## 🧪 Theme 2: Small Language Models for On‑Device or Edge Scenarios

**Why it fits you:** Highly algorithmic – you will work with constrained environments (memory, latency, CPU‑only). You can design and run systematic experiments comparing speed/accuracy trade‑offs, which is very compatible with a competitive‑programming mindset.

| Aspect | Details |
|--------|---------|
| Core research question | For a given task (e.g., intent classification, email auto‑completion, or a simple game assistant), how small can a model be while maintaining acceptable performance, and which architectural choices matter most? |
| Novice‑friendly aspect | SLMs (sub‑billion to 3B parameters) can run fully offline on a laptop or a Raspberry Pi. There are open‑source SLM families (Microsoft Phi‑3 mini, Qwen2.5‑0.5B/1.5B, TinyLlama, Gemma 2 2B). You can measure CPU inference time, memory footprint, and accuracy. |
| 9‑month roadmap | **Month 1:** Select a task and collect/curate a small dataset (e.g., 5k–10k examples). **Months 2–3:** Benchmark 3–4 SLMs (0.5B, 1.5B, 3B) on that task using zero‑shot prompting. **Months 4–6:** Apply 4‑bit or 8‑bit quantisation (using llama.cpp, or Hugging Face `bitsandbytes`) and measure the accuracy drop vs. latency/memory gain. **Months 7–8:** If time permits, fine‑tune the best‑performing quantised model using LoRA on a low‑end device. **Month 9:** Document the pipeline, the benchmarking methodology, and the trade‑off analysis. |
| Potential extensions | Implement **model cascading**: route easy queries to a tiny SLM (0.5B) and hard ones to a slightly larger model (3B), measuring the overall cost savings. This is an optimisation problem you already know how to solve. |

---

## 🧪 Theme 3: Prompt Compression for Cost‑Efficient LLM Inference

**Why it fits you:** Extremely algorithmic – you will implement a token‑level compressor, measure compression ratio vs. performance degradation, and build a small evaluation benchmark. Many compression techniques use **attention‑based selection**, **dictionary‑encoding**, or **token‑attribution pruning**, all of which are deterministic and implementable in a few hundred lines of Python.

| Aspect | Details |
|--------|---------|
| Core research question | How much can we compress input prompts (for a specific task like summarisation or question answering) before output quality significantly degrades, and what is the latency reduction? |
| Novice‑friendly aspect | You can start with a simple **baseline compressor** (e.g., keep only top‑k tokens by TF‑IDF or by a small‑model attention score). Then compare with a more advanced method (e.g., DAC – dynamic attention‑aware compression). You only need access to an open LLM (e.g., Llama 3 8B via Hugging Face or a local Ollama server). |
| 9‑month roadmap | **Month 1:** Study the prompt compression literature and collect two datasets: summarisation (CNN/DailyMail) and question answering (SQuAD or Natural Questions). **Months 2–3:** Implement a trivial baseline (truncation to first k tokens) and a simple attention‑based compressor. **Months 4–5:** Evaluate different compression ratios (50%, 70%, 90%) and measure ROUGE/BLEU (summarisation) or exact‑match (QA) vs. inference latency. **Months 6–7:** Implement a more advanced compressor from a recent paper (e.g., FrugalPrompt or Local‑Splitter) and compare against your baselines. **Month 8:** Run statistical significance tests and create plots of compression ratio vs. performance. **Month 9:** Write up the comparative study, focusing on which compression method works best for which task. |
| Potential extensions | Build a small **Python library** for prompt compression that others can plug into their LLM pipelines. This is a concrete deliverable that looks great on a CV. |

---

## 🧪 Theme 4: Model Merging (Model Soups) for Skill Composition

**Why it fits you:** This is essentially **numerical linear algebra** combined with empirical experimentation. You take multiple fine‑tuned LoRA adapters (or full model checkpoints) and combine them via weighted averaging. The core operation is simple (vector addition and scalar multiplication), yet the behaviour is non‑trivial.

| Aspect | Details |
|--------|---------|
| Core research question | Can simple arithmetic merging of LoRA adapters (each specialised for a different task) produce a single adapter that performs well on a composite task, without any additional training? |
| Novice‑friendly aspect | You do not need to train large models. Instead, you fine‑tune two or three small LoRA adapters on distinct tasks (e.g., sentiment analysis, question classification, and text simplification). Then you merge them using linear interpolation or more advanced weighted averaging (e.g., SoCE – Soup of Category Experts) and evaluate on a task that requires multiple skills (e.g., answer a question about sentiment). |
| 9‑month roadmap | **Month 1:** Understand model merging literature (linear weight soups, task vectors, Fisher‑weighted averaging, DARE). **Months 2–3:** Fine‑tune 3 LoRA adapters on 3 different tasks using a small base model (e.g., Llama 3.2 3B). **Months 4–5:** Implement merging strategies (uniform average, weighted by validation performance, Fisher‑weighted). **Months 6–7:** Evaluate merged models on held‑out composite tasks. Compare against an ensemble (which runs all three adapters separately and combines outputs). Measure inference cost (the merged model is a single forward pass, much cheaper than an ensemble). **Month 8:** Analyse the correlation between task similarity and mergability. **Month 9:** Write a report with clear trade‑off analysis. |
| Potential extensions | Experiment with **LoRA soups** (merging multiple LoRA adapters trained for the same task but with different hyperparameters or seeds). This can improve robustness without extra inference cost. |

---

## 🧪 Theme 5: Synthetic Data Generation and Distillation for Knowledge Transfer

**Why it fits you:** This theme sits at a fascinating intersection of algorithm design and empirical experimentation. You will generate a dataset using a large teacher model, then train a small student model on that synthetic data, measuring how well the student "distills" the teacher's capabilities. The core challenge is avoiding **model collapse** (where iterative self‑training degrades performance).

| Aspect | Details |
|--------|---------|
| Core research question | What is the optimal ratio of synthetic data to real data when distilling a large teacher model into a tiny student model for a specific reasoning or domain‑specific task? |
| Novice‑friendly aspect | You can run the entire pipeline using a free Google Colab GPU for the student training (e.g., train a TinyLlama or Phi‑3 mini on a few thousand synthetic examples). The teacher can be a larger open model (Llama 3 70B via Together.ai’s free tier, or GPT‑3.5 Turbo with a small budget). You only need API access for a limited number of calls (e.g., 10k generations). |
| 9‑month roadmap | **Month 1:** Choose a task with clear evaluation (e.g., grade‑school math reasoning (GSM8K) or a simple code‑generation benchmark (HumanEval)). **Month 2:** Generate a synthetic dataset by prompting the teacher model with 500–1000 seed examples (augment each seed to produce multiple variations). **Months 3–4:** Train a small student model (e.g., 0.5B–1.5B) on different mixtures: 100% synthetic, 50% synthetic + 50% real, 100% real (baseline). **Months 5–6:** Evaluate on held‑out real test data. Measure accuracy, training time, and how much the student improves over a zero‑shot baseline. **Month 7:** Investigate **model collapse**: if you iteratively train on synthetic outputs, does accuracy degrade? **Month 8:** Analyse the diversity of synthetic generations (e.g., lexical diversity, correct vs. incorrect reasoning patterns). **Month 9:** Write a case study comparing distillation performance across different student sizes. |
| Potential extensions | Use **context‑free synthetic data** to mitigate catastrophic forgetting when fine‑tuning on a narrow domain. This is a highly relevant research direction in continual learning for LLMs. |

---

## How to Choose Among These Themes

| Theme | Hardware Requirement | Code‑first? | CP Skill Leverage | Ease of Publication‑style Report |
|-------|----------------------|-------------|-------------------|----------------------------------|
| 1. PEFT for Domain | Low–Med (6‑12 GB VRAM) | ✅ High (LoRA config sweeps) | Medium (algorithmic understanding of low‑rank matrices) | ✅ High (clear hyperparameter sweeps, trade‑off curves) |
| 2. SLMs on Edge | Low (CPU) | ✅ Very high (benchmarking, quantisation, cascading) | ✅ Very high (scheduling, resource allocation) | ✅ High (engineering focus, reproducibility) |
| 3. Prompt Compression | Low–Med (any LLM via API or local) | ✅ High (implement token pruning) | ✅ Very high (implementing deterministic algorithms) | ✅ High (comparative evaluation) |
| 4. Model Merging | Low (only inference of small models) | Medium (more experimentation than coding) | Medium (linear algebra) | Medium–High (requires careful experimental design) |
| 5. Synthetic Data & Distillation | Low (student training on Colab) | Medium (data generation pipeline + training) | Low–Medium (focus on experiment design) | ✅ High (hot research topic, easy to frame as a case study) |