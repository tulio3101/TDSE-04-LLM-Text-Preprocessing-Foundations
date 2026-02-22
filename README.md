# LLM Text Preprocessing Foundations

This project implements the core text preprocessing pipeline for Large Language Models (LLMs) based on Chapter 2 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka. The notebook demonstrates how to prepare text data for training LLMs, covering tokenization, encoding, embeddings, and data sampling techniques.

## Getting Started

These instructions will help you set up the project and run the notebook locally for learning and experimentation purposes.

### Prerequisites

What you need to install the software:

- Python 3.9 or higher
- PyTorch
- tiktoken (OpenAI's BPE tokenizer)

### Installing

A step by step guide to get a development environment running:

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd TDSE-04-LLM-Text-Preprocessing-Foundations
   ```

2. Install the required dependencies:
   ```bash
   pip install torch tiktoken
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook embeddings.ipynb
   ```

## Project Structure

- **embeddings.ipynb**: Main notebook containing all the code and explanations
- **the-verdict.txt**: Sample text dataset (short story by Edith Wharton)

## What You Will Learn

This notebook covers the essential preprocessing steps for LLMs:

### 1. Tokenization
Breaking text into smaller units (words, subwords, characters) using regular expressions and BytePair Encoding (BPE).

### 2. Token Encoding
Converting tokens into numerical IDs using a vocabulary dictionary.

### 3. Special Context Tokens
Adding special tokens like `<|endoftext|>` and `<|unk|>` for handling unknown words and text boundaries.

### 4. Data Sampling with Sliding Window
Creating training samples using a sliding window approach with configurable `max_length` and `stride`.

### 5. Token Embeddings
Converting token IDs into dense vector representations using embedding layers.

### 6. Positional Encodings
Adding position information to token embeddings so the model can understand word order.

## Experiment: Sliding Window Parameters

The notebook includes an experiment to understand how `max_length` and `stride` affect the number of training samples:

```python
# Example: max_length=4, stride=1
# More overlap, more samples, better context learning

# Example: max_length=4, stride=4
# No overlap, fewer samples, faster training
```

**Why overlap is useful:**
- Smaller stride → more overlapping samples → model sees more contextual variations
- Larger stride → less overlap → faster training but potentially less context

## Built With

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's BPE tokenizer
- [Jupyter](https://jupyter.org/) - Interactive computing environment

## References

- [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
- [Original notebook](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02) from the author
- [tiktoken GitHub](https://github.com/openai/tiktoken)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
