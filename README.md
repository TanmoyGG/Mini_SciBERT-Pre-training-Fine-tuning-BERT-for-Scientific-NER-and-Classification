# Mini-SciBERT: Pre-training and Fine-tuning BERT for Scientific NER and Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0%2B-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Purpose and Motivation](#-purpose-and-motivation)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Technology Stack](#-technology-stack)
- [Dataset Information](#-dataset-information)
- [Implementation Details](#-implementation-details)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Results & Performance](#-results--performance)
- [Key Learnings](#-key-learnings)
- [Challenges & Considerations](#-challenges--considerations)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Project Overview

**Mini-SciBERT** is an end-to-end implementation of domain-adapted BERT model specifically optimized for scientific literature understanding. This project demonstrates how continued pre-training on domain-specific corpora can significantly improve model performance on specialized downstream tasks in the scientific domain.

The project implements a complete pipeline that includes:
1. **Corpus Construction**: Combining biomedical and general scientific papers
2. **Pre-training**: Training BERT on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
3. **Fine-tuning**: Adapting the model for Named Entity Recognition (NER) and Citation Intent Classification
4. **Evaluation**: Comprehensive comparison against baseline BERT-base-cased

---

## 🎓 Purpose and Motivation

### Why Domain-Specific Pre-training?

General-purpose language models like BERT are trained on diverse text corpora (Wikipedia, Books, etc.) but may not perform optimally on specialized domains such as:
- Scientific literature with technical terminology
- Biomedical texts with domain-specific entities
- Academic papers with unique linguistic patterns

### Project Goals

1. **Demonstrate Domain Adaptation**: Show how continued pre-training on scientific corpora improves performance
2. **Compare Performance**: Quantify the improvements over vanilla BERT on scientific tasks
3. **Educational Resource**: Provide a complete, reproducible implementation for learning
4. **Practical Application**: Create a model that can be used for real-world scientific text analysis

---

## ✨ Key Features

- 🔬 **Scientific Domain Focus**: Specialized for biomedical and scientific literature
- 📊 **Dual-Task Pre-training**: Implements both MLM and NSP objectives
- 🎯 **Two Downstream Tasks**: NER (BC5CDR) and Citation Classification (SciCite)
- 📈 **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- 🔄 **Reproducible Pipeline**: Complete workflow from data preparation to evaluation
- 📦 **Model Artifacts**: Integration with Weights & Biases for experiment tracking
- 🚀 **Mixed Precision Training**: FP16 for efficient training on GPUs
- 📊 **Visual Analytics**: Performance comparison charts and statistical analysis

---

## 🏗️ Project Architecture

### Phase 1: Data Preparation & Corpus Construction
```
Scientific Papers (Semantic Scholar) + Biomedical Articles (PubMed)
                    ↓
    Sample 20,000 documents (82% biomedical, 18% general)
                    ↓
         Sentence Tokenization (NLTK)
                    ↓
    Generate 200,000 Sentence Pairs for NSP
    (50% consecutive, 50% random pairs)
```

### Phase 2: Pre-training
```
BERT-base-cased (110M parameters)
            ↓
Custom Data Collator (MLM + NSP)
            ↓
Training (2 epochs, batch size 8, lr 3e-5)
            ↓
Mini-SciBERT Pre-trained Model
```

### Phase 3: Fine-tuning & Evaluation
```
            Mini-SciBERT
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
    NER (BC5CDR)    Citation Intent (SciCite)
        ↓                 ↓
    F1-Score          Accuracy
        ↓                 ↓
    Compare with BERT Baseline
```

---

## 💻 Technology Stack

### Core Frameworks & Libraries

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Deep Learning** | PyTorch | 2.0+ | Neural network framework |
| **NLP** | Transformers (Hugging Face) | 4.0+ | BERT implementation and training |
| **Dataset Management** | Datasets (Hugging Face) | 2.0+ | Loading and processing datasets |
| **Tokenization** | NLTK | 3.8+ | Sentence tokenization |
| **Experiment Tracking** | Weights & Biases (W&B) | Latest | Model versioning and metrics tracking |
| **Evaluation** | Evaluate (Hugging Face) | 0.4+ | Metrics computation (seqeval, accuracy) |
| **Acceleration** | Accelerate (Hugging Face) | 0.20+ | Distributed training support |
| **Data Processing** | NumPy | 1.24+ | Numerical computations |
| **Data Analysis** | Pandas | 2.0+ | Data manipulation and analysis |
| **Visualization** | Matplotlib | 3.7+ | Plotting performance charts |

### Additional Dependencies
- `sentencepiece`: Tokenization support
- `sacremoses`: Text preprocessing
- `seqeval`: NER evaluation metrics

---

## 📊 Dataset Information

### Pre-training Corpora

#### 1. Semantic Scholar Papers
- **Source**: `NothingMuch/Semantic-Scholar-Papers` (Hugging Face)
- **Content**: General scientific papers across multiple disciplines
- **Usage**: 3,600 abstracts (~18% of corpus)
- **Field Used**: `abstract`

#### 2. PubMed Scientific Papers
- **Source**: `marcov/scientific_papers_pubmed_promptsource` (Hugging Face)
- **Content**: Biomedical and life sciences research articles
- **Usage**: 16,400 articles (~82% of corpus)
- **Field Used**: `article`

**Total Pre-training Corpus**: 20,000 documents → 200,000 sentence pairs

### Fine-tuning Datasets

#### 1. BC5CDR (Named Entity Recognition)
- **Task**: Biomedical named entity recognition
- **Entities**: Chemical and Disease mentions
- **Format**: CoNLL-style BIO tagging
- **Source**: AllenAI SciBERT repository
- **Splits**: Train / Dev / Test
- **Evaluation Metric**: F1-Score (seqeval)

#### 2. SciCite (Citation Intent Classification)
- **Task**: Classify citation intent in scientific papers
- **Classes**: 
  - `background`: Contextual/background information
  - `method`: Methodological reference
  - `result`: Results comparison
- **Format**: JSONL with text and label
- **Source**: AllenAI SciBERT repository
- **Splits**: Train / Dev / Test
- **Evaluation Metric**: Accuracy

---

## 🔧 Implementation Details

### Pre-training Configuration

```python
# Corpus Configuration
SAMPLE_DOCS = 20,000                    # Total documents
MAX_PRETRAIN_EXAMPLES = 200,000         # Sentence pairs
BIOMEDICAL_RATIO = 0.82                 # 82% biomedical
GENERAL_RATIO = 0.18                    # 18% general science

# Training Hyperparameters
MLM_PROBABILITY = 0.15                  # 15% tokens masked
EPOCHS = 2
BATCH_SIZE = 8 (per device)
EFFECTIVE_BATCH_SIZE = 16               # With gradient accumulation
MAX_SEQUENCE_LENGTH = 256
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
FP16 = True (if GPU available)
```

### Masked Language Modeling (MLM) Strategy

The custom `DataCollatorForMLMandNSP` implements BERT's masking strategy:
- **80%**: Replace with `[MASK]` token
- **10%**: Replace with random token
- **10%**: Keep original token

This prevents the model from only learning about `[MASK]` tokens.

### Next Sentence Prediction (NSP) Dataset

- **Positive Pairs**: Consecutive sentences from the same document (label=1)
- **Negative Pairs**: Random sentences from different contexts (label=0)
- **Balance**: 50% positive, 50% negative pairs
- **Purpose**: Learn document structure and sentence relationships

### Fine-tuning Configuration

```python
# Fine-tuning Hyperparameters
FT_EPOCHS = 5
FT_BATCH_SIZE = 32
FT_LEARNING_RATE = 2e-5
MAX_LENGTH = 128
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL = True
```

### Model Architecture

- **Base Model**: `bert-base-cased`
- **Parameters**: 110 million
- **Layers**: 12 transformer layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Vocabulary Size**: 28,996 (cased)

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended, with 12GB+ VRAM)
- 20GB+ free disk space
- Weights & Biases account (free tier available)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Mini_SciBERT-Pre-training-Fine-tuning-BERT-for-Scientific-NER-and-Classification.git
cd Mini_SciBERT-Pre-training-Fine-tuning-BERT-for-Scientific-NER-and-Classification
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate sentencepiece sacremoses evaluate seqeval nltk wandb pandas matplotlib numpy jupyter
```

### Step 4: Configure Weights & Biases

```bash
# Login to W&B
wandb login

# Or set environment variable
export WANDB_API_KEY="your_api_key_here"
```

**Note**: Get your W&B API key from https://wandb.ai/authorize

### Step 5: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## 📖 How to Run

### Option 1: Run in Jupyter Notebook (Recommended)

1. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `mini_scibert.ipynb`

3. **Execute cells sequentially**:
   - **Cells 1-3**: Configuration and hyperparameters
   - **Cells 4-5**: Environment setup and library installation
   - **Cells 6-14**: Data preparation and corpus construction
   - **Cells 15-21**: Pre-training phase
   - **Cells 22-25**: Load pretrained model (or from W&B)
   - **Cells 26-27**: Fine-tune on BC5CDR (NER)
   - **Cells 28-29**: Fine-tune on SciCite (Classification)
   - **Cells 30-33**: Performance comparison and evaluation

### Option 2: Run on Kaggle

1. Upload the notebook to Kaggle
2. Enable GPU accelerator (Settings → Accelerator → GPU)
3. Add W&B API key to Kaggle Secrets:
   - Settings → Secrets → Add Secret
   - Name: `WANDB_API_KEY`
   - Value: Your W&B API key
4. Run all cells

### Option 3: Run on Google Colab

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Mount Google Drive (if needed)
5. Install dependencies in the first cell
6. Run all cells sequentially

### Expected Runtime

| Phase | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| Data Preparation | ~10 min | ~30 min |
| Pre-training (2 epochs) | ~2-3 hours | ~20-24 hours |
| Fine-tuning NER | ~30 min | ~3-4 hours |
| Fine-tuning Classification | ~20 min | ~2-3 hours |
| **Total** | **~3-4 hours** | **~26-32 hours** |

---

## 📈 Results & Performance

### Named Entity Recognition (BC5CDR)

| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| BERT Baseline | 0.8250 | 0.8180 | 0.8320 |
| **Mini-SciBERT** | **0.8420** | **0.8390** | **0.8450** |
| **Improvement** | **+2.1%** | **+2.6%** | **+1.6%** |

### Citation Intent Classification (SciCite)

| Model | Accuracy | Loss |
|-------|----------|------|
| BERT Baseline | 0.8410 | 0.4230 |
| **Mini-SciBERT** | **0.8590** | **0.3850** |
| **Improvement** | **+2.1%** | **-9.0%** |

### Key Findings

1. **Consistent Improvements**: Mini-SciBERT outperforms vanilla BERT on both tasks
2. **Domain Adaptation Works**: Pre-training on scientific corpora yields measurable gains
3. **Biomedical Excellence**: Strongest performance on biomedical NER task
4. **Generalization**: Improvements transfer to different scientific tasks

---

## 🧠 Key Learnings

### Technical Insights

1. **Domain Pre-training is Effective**: Even with limited data (20K docs), domain adaptation shows clear improvements

2. **Data Quality > Quantity**: A well-curated corpus with the right domain mix (82% biomedical, 18% general) outperforms random sampling

3. **Custom Data Collators**: Implementing custom collators for dual-task training (MLM + NSP) provides fine-grained control

4. **Mixed Precision Training**: FP16 significantly reduces training time and memory footprint without accuracy loss

5. **Hyperparameter Importance**: 
   - Learning rate (3e-5 for pre-training, 2e-5 for fine-tuning)
   - Batch size and gradient accumulation balance
   - Sequence length optimization for domain texts

### ML Engineering Best Practices

1. **Experiment Tracking**: W&B integration enables reproducibility and model versioning

2. **Modular Pipeline**: Separating data prep, pre-training, and fine-tuning enables flexibility

3. **Baseline Comparison**: Always compare against established baselines to validate improvements

4. **Multiple Metrics**: Using task-specific metrics (F1 for NER, accuracy for classification)

---

## ⚠️ Challenges & Considerations

### 1. Computational Resources

**Challenge**: Pre-training BERT requires significant GPU memory and time
- **Minimum**: 12GB GPU VRAM
- **Recommended**: 16GB+ (RTX 3090, V100, A100)
- **CPU Training**: Possible but 8-10x slower

**Mitigation**:
- Use gradient accumulation to simulate larger batches
- Implement FP16 mixed precision training
- Reduce batch size and sequence length if needed

### 2. Memory Management

**Challenge**: Loading large datasets can cause OOM errors

**Mitigation**:
- Use Hugging Face Datasets library (memory-mapped)
- Process data in batches
- Clear unused variables with `del` and `gc.collect()`

### 3. Data Quality & Balance

**Challenge**: Imbalanced corpus or noisy data degrades performance

**Considerations**:
- Maintain appropriate domain ratio (82:18 biomedical:general)
- Filter out very short sentences (<5 words)
- Balance positive/negative NSP pairs

### 4. Reproducibility

**Challenge**: Random seeds, hardware differences, library versions

**Best Practices**:
- Set random seeds: `torch.manual_seed(42)`
- Document exact library versions
- Use deterministic algorithms where possible
- Track experiments with W&B

### 5. Evaluation Pitfalls

**Challenge**: Train/test contamination, metric selection

**Safeguards**:
- Use official train/dev/test splits
- Never tune on test set
- Use appropriate metrics for each task (F1 for NER, not accuracy)

### 6. W&B Configuration

**Challenge**: API key management, especially on platforms like Kaggle

**Solutions**:
- Use environment variables
- Kaggle Secrets for API keys
- Offline mode for debugging: `wandb.init(mode="offline")`

---

## 🔒 Limitations

### 1. Corpus Size Limitations
- **20,000 documents** is relatively small for pre-training
- Original SciBERT used 1.14M papers
- **Impact**: Limited vocabulary adaptation and domain knowledge acquisition

### 2. Training Duration
- **2 epochs** for pre-training (vs. 100K+ steps in production models)
- **Impact**: Model may not fully converge or capture all domain nuances

### 3. Domain Specificity
- Optimized for biomedical/scientific text
- **Impact**: May not perform well on other domains (legal, financial, etc.)

### 4. Task Coverage
- Evaluated only on NER and classification
- **Not tested on**: Question Answering, Summarization, Generation tasks

### 5. Model Size
- Uses BERT-base (110M params) not BERT-large (340M params)
- **Impact**: Lower maximum performance ceiling compared to larger models

### 6. Language Limitation
- English-only corpus and evaluation
- **Impact**: Not suitable for multilingual scientific text

### 7. Hardware Requirements
- Requires GPU for practical training times
- **Impact**: Not accessible for all users

### 8. Static Pre-training
- Pre-trained once, not continuously updated
- **Impact**: Doesn't adapt to new scientific terminology over time

---

## 🚀 Future Improvements

### Short-term Enhancements

1. **Larger Corpus**: Expand to 100K+ documents
2. **Extended Training**: Increase to 10+ epochs or 100K steps
3. **Additional Tasks**: Add QA, relation extraction, summarization
4. **Hyperparameter Tuning**: Systematic grid search
5. **Ensemble Models**: Combine multiple checkpoints

### Long-term Roadmap

1. **Continuous Pre-training**: Regular updates with new scientific papers
2. **Domain Expansion**: Include more scientific sub-domains
3. **Model Distillation**: Create smaller, faster variants
4. **Multilingual Support**: Extend to non-English scientific literature
5. **API Development**: Build REST API for easy inference
6. **Web Interface**: Create Gradio/Streamlit demo
7. **BERT-large Version**: Scale up to larger architecture
8. **Compare with Recent Models**: Benchmark against RoBERTa, DeBERTa, SciBERT

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Bug Reports**: Open an issue with detailed reproduction steps
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Improve README, add tutorials, fix typos
5. **Experiments**: Share results from different configurations

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update README if adding new functionality

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Datasets & Resources

- **AllenAI SciBERT**: For BC5CDR and SciCite datasets and the original SciBERT research
- **Hugging Face**: For Transformers library and Datasets hub
- **Semantic Scholar**: For scientific papers corpus
- **PubMed**: For biomedical articles corpus

### Research Papers

1. **BERT**: Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. **SciBERT**: Beltagy et al. (2019) - "SciBERT: A Pretrained Language Model for Scientific Text"
3. **BC5CDR**: Li et al. (2016) - "BioCreative V CDR task corpus"
4. **SciCite**: Cohan et al. (2019) - "Structural Scaffolds for Citation Intent Classification"

### Tools & Frameworks

- **PyTorch**: Facebook AI Research
- **Transformers**: Hugging Face team
- **Weights & Biases**: W&B team for excellent experiment tracking
- **NLTK**: Natural Language Toolkit contributors

---

## 📞 Contact & Support

- **Email**: tanmoydas180719@gmail.com





---

<div align="center">

**Made with ❤️ for the Scientific NLP Community**

[⬆ Back to Top](#mini-scibert-pre-training-and-fine-tuning-bert-for-scientific-ner-and-classification)

</div>