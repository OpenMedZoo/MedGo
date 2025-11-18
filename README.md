---
license: apache-2.0
language:
- zh
- en
metrics:
- accuracy
base_model:
- Qwen/Qwen3-32B
pipeline_tag: text-generation
library_name: transformers
tags:
- medical
model-index:
- name: Med-Go-32B
  results:
  - task:
      type: text-generation
    dataset:
      type: medical_eval_hle
      name: Medical-Eval-HLE
    metrics:
    - name: accuracy
      type: accuracy
      value: 19.4
      verified: false
  - task:
      type: text-generation
    dataset:
      type: supergpqa
      name: SuperGPQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 37.2
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medbullets
      name: Medbullets
    metrics:
    - name: accuracy
      type: accuracy
      value: 57.8
      verified: false
  - task:
      type: text-generation
    dataset:
      type: mmlu_pro
      name: MMLU-pro
    metrics:
    - name: accuracy
      type: accuracy
      value: 64.3
      verified: false
  - task:
      type: text-generation
    dataset:
      type: afrimedqa
      name: AfrimedQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 74.7
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medmcqa
      name: MedMCQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 68.3
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medqa_usmle
      name: MedQA-USMLE
    metrics:
    - name: accuracy
      type: accuracy
      value: 76.8
      verified: false
  - task:
      type: text-generation
    dataset:
      type: cmb
      name: CMB
    metrics:
    - name: accuracy
      type: accuracy
      value: 92.5
      verified: false
  - task:
      type: text-generation
    dataset:
      type: cmexam
      name: CMExam
    metrics:
    - name: accuracy
      type: accuracy
      value: 87.4
      verified: false
  - task:
      type: text-generation
    dataset:
      type: pubmedqa
      name: PubMedQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 76.6
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medexqa
      name: MedExQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 81.5
      verified: false
  - task:
      type: text-generation
    dataset:
      type: explaincpe
      name: ExplainCPE
    metrics:
    - name: accuracy
      type: accuracy
      value: 89.5
      verified: false
  - task:
      type: text-generation
    dataset:
      type: mmlu_med
      name: MMLU-Med
    metrics:
    - name: accuracy
      type: accuracy
      value: 87.4
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medxperqa
      name: MedXperQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 20.7
      verified: false
  - task:
      type: text-generation
    dataset:
      type: anesbench
      name: AnesBench
    metrics:
    - name: accuracy
      type: accuracy
      value: 53.1
      verified: false
  - task:
      type: text-generation
    dataset:
      type: diagnosisarena
      name: DiagnosisArena
    metrics:
    - name: accuracy
      type: accuracy
      value: 64.4
      verified: false
  - task:
      type: text-generation
    dataset:
      type: clinbench_hbp
      name: Clinbench-HBP
    metrics:
    - name: accuracy
      type: accuracy
      value: 80.6
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medpair
      name: MedPAIR
    metrics:
    - name: accuracy
      type: accuracy
      value: 32.3
      verified: false
  - task:
      type: text-generation
    dataset:
      type: amqa
      name: AMQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 72.7
      verified: false
  - task:
      type: text-generation
    dataset:
      type: medethicaleval
      name: MedethicalEval
    metrics:
    - name: accuracy
      type: accuracy
      value: 92.2
      verified: false
---

# MedGo: Medical Large Language Model Based on Qwen3-32B

<div align="center">

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/OpenMedZoo/MedGo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)


English | [简体中文](./README_CN.md)

</div>

## 📋 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Training Details](#training-details)
- [Use Cases](#use-cases)
- [Limitations & Risks](#limitations--risks)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## 🎯 Introduction

**MedGo** is a general-purpose medical large language model fine-tuned from **Qwen3-32B**, designed for clinical medicine and research scenarios. The model is trained on large-scale multi-source medical corpora and enhanced with complex case data, supporting various capabilities including medical Q&A, clinical summary, clinical reasoning, multi-turn dialogue, and scientific text generation.

### 🌟 Core Capabilities

- **📚 Medical Knowledge Q&A**: Professional responses based on authoritative medical literature and clinical guidelines
- **📝 Clinical Documentation**: Automated medical record summaries, diagnostic reports, and medical documentation
- **🔍 Clinical Reasoning**: Differential diagnosis, examination recommendations, and treatment suggestions
- **💬 Multi-turn Dialogue**: Patient-doctor interaction simulation and complex case discussions
- **🔬 Research Support**: Literature summarization, research idea generation, and quality control review

## ✨ Key Features

| Feature | Details |
|---------|---------|
| **Base Architecture** | Qwen3-32B |
| **Parameters** | 32B |
| **Domain** | Clinical Medicine, Research Support, Healthcare System Integration |
| **Fine-tuning Method** | SFT + Preference Alignment (DPO/KTO) |
| **Data Sources** | Authoritative medical literature, clinical guidelines, real cases (anonymized) |
| **Deployment** | Local deployment, HIS/EMR system integration |
| **License** | Apache 2.0 |

## 📊 Performance

MedGo demonstrates excellent performance across multiple medical and general evaluation benchmarks, showing competitive results among 32B-parameter models:

### Key Benchmark Results

- **AIMedQA**: Medical question answering comprehension
- **CME**: Clinical reasoning evaluation
- **DiagnosisArena**: Diagnostic capability assessment
- **MedQA / MedMCQA**: Medical multiple-choice questions
- **PubMedQA**: Biomedical literature Q&A
- **MMLU-Pro**: Comprehensive capability evaluation

![Performance Comparison](./main_results.png)

**Performance Highlights**:
- ✅ **Average Score**: ~70 points (excellent performance in the 32B parameter class)
- ✅ **Strong Tasks**: Clinical reasoning (DiagnosisArena, CME) and multi-turn medical Q&A
- ✅ **Balanced Capability**: Good performance in medical semantic understanding and multi-task generalization


## 🚀 Quick Start

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.35.0
- CUDA >= 11.8 (for GPU inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/OpenMedZoo/MedGo.git
cd MedGo

# Install dependencies
pip install -r requirements.txt
```

### Model Download

Download model weights from HuggingFace:

```bash
# Using huggingface-cli
huggingface-cli download OpenMedZoo/MedGo --local-dir ./models/MedGo

# Or using git-lfs
git lfs install
git clone https://huggingface.co/OpenMedZoo/MedGo
```

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = "OpenMedZoo/MedGo"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)

# Medical Q&A example
messages = [
    {"role": "system", "content": "You are a professional medical assistant. Please answer questions based on medical knowledge."},
    {"role": "user", "content": "What is hypertension and what are the common treatment methods?"}
]

# Generate response
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(response)
```

### Batch Inference

```bash
# Use the provided inference script
python scripts/inference.py \
    --model_path OpenMedZoo/MedGo \
    --input_file examples/medical_qa.jsonl \
    --output_file results/predictions.jsonl \
    --batch_size 4
```

### Accelerated Inference with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(model="OpenMedZoo/MedGo", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# Batch inference
prompts = [
    "What are the symptoms and treatment methods for diabetes?",
    "What dietary precautions should hypertensive patients take?"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

## 🔧 Training Details

MedGo employs a **two-stage fine-tuning strategy** to balance general medical knowledge with clinical task adaptation.

### Stage I: General Medical Alignment

**Objective**: Establish a solid foundation of medical knowledge and improve Q&A standardization

- **Data Sources**:
  - Authoritative medical literature (PubMed, medical textbooks)
  - Clinical guidelines and diagnostic standards
  - Medical encyclopedia entries and terminology databases
  
- **Training Methods**:
  - Supervised Fine-Tuning (SFT)
  - Chain-of-Thought (CoT) guided samples
  - Medical terminology alignment and safety constraints

### Stage II: Clinical Task Enhancement

**Objective**: Enhance complex case reasoning and multi-task processing capabilities

- **Data Sources**:
  - Real medical records (fully anonymized)
  - Outpatient and emergency records with complex multi-diagnosis samples
  - Research articles and quality control cases
  
- **Data Augmentation Techniques**:
  - Semantic paraphrasing and multi-perspective expansion
  - Complex case synthesis
  - Doctor-patient interaction simulation
  
- **Training Methods**:
  - Multi-Task Learning (medical record summary, differential diagnosis, examination suggestions, etc.)
  - Preference Alignment (DPO/KTO)
  - Expert feedback iterative optimization

### Training Optimization Focus

- ✅ Strengthen information extraction and cross-evidence reasoning for complex cases
- ✅ Improve medical consistency and interpretability of outputs
- ✅ Optimize expression compliance and safety
- ✅ Continuous iteration through expert samples and automated evaluation

## 💡 Use Cases

### ✅ Suitable Scenarios

| Scenario | Description |
|----------|-------------|
| **Clinical Assistance** | Preliminary diagnosis suggestions, medical record writing, formatted report generation |
| **Research Support** | Literature summarization, research idea generation, data analysis assistance |
| **Quality Control** | Medical document compliance checking, clinical process quality control |
| **System Integration** | Embedded in HIS/EMR systems to provide intelligent decision support |
| **Medical Education** | Case discussions, medical knowledge Q&A, clinical reasoning training |

### 🚫 Unsuitable Scenarios

- ❌ **Cannot Replace Doctors**: Only an auxiliary tool, not a standalone diagnostic basis
- ❌ **High-Risk Operations**: Not recommended for surgical decisions or other high-risk medical operations
- ❌ **Rare Disease Limitations**: May perform poorly on rare diseases outside training data
- ❌ **Emergency Care**: Not suitable for scenarios requiring immediate decisions

## ⚠️ Limitations & Risks

### Model Limitations

1. **Understanding Bias**: Despite covering extensive medical knowledge, may still produce understanding biases or incorrect recommendations
2. **Complex Cases**: Higher risk for cases with complex conditions, severe complications, or missing information
3. **Knowledge Currency**: Medical knowledge continuously updates; training data may lag
4. **Language Limitation**: Primarily designed for Chinese medical scenarios; performance in other languages may vary

### Usage Recommendations

- ⚠️ Use in controlled environments with clinical expert review of generated results
- ⚠️ Treat model outputs as auxiliary references, not final diagnostic conclusions
- ⚠️ For sensitive cases or high-risk scenarios, expert consultation is mandatory
- ⚠️ Deployment requires internal validation, security review, and clinical testing

### Data Privacy & Compliance

- 🔒 Training data fully anonymized
- 🔒 Attention to patient privacy protection during use
- 🔒 Production deployment must comply with healthcare data security regulations (e.g., HIPAA, GDPR)
- 🔒 Local deployment recommended to avoid sensitive data transmission

## 📚 Citation

If MedGo is helpful for your research or project, please cite our work:

```bibtex
@misc{openmedzoo_2025,
	author       = { OpenMedZoo },
	title        = { MedGo (Revision 640a2e2) },
	year         = 2025,
	url          = { https://huggingface.co/OpenMedZoo/MedGo },
	doi          = { 10.57967/hf/7024 },
	publisher    = { Hugging Face }
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

**Commercial Use Notice**:
- ✅ Commercial use and modification allowed
- ✅ Original license and copyright notice must be retained
- ✅ Contact us for technical support when integrating into healthcare systems

## 🤝 Contributing

We welcome community contributions! Here's how to participate:

### Contribution Types

- 🐛 Submit bug reports
- 💡 Propose new features
- 📝 Improve documentation
- 🔧 Submit code fixes or optimizations
- 📊 Share evaluation results and use cases


## 🙏 Acknowledgments

Thanks to all contributors to the MedGo project:

- Model development and fine-tuning algorithm team
- Data annotation and quality control team
- Clinical expert guidance and review team
- Open-source community support and feedback

Special thanks to:
- [Qwen Team](https://github.com/QwenLM/Qwen) for providing excellent foundation models
- All healthcare institutions that provided data and feedback

## 📧 Contact

- **HuggingFace**: [Model Homepage](https://huggingface.co/OpenMedZoo/MedGo)

---

<div align="center">
</div>