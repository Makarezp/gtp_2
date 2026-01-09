# GPT-2 from Scratch

A from-scratch implementation of GPT-2 following [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY).

## Features

- Complete GPT-2 architecture (12 layers, 12 heads, 768 dims, 124M params)
- Weight tying (input embeddings + output head)
- Proper weight initialization with scaled projection layers
- Custom DataLoader for streaming text data

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place your training data in `input.txt`, then:

```bash
python train_gpt2.py
```

**Expected Output:**
```
step 0, loss: 11.05
step 20, loss: 0.35
step 49, loss: 0.005
```

## Key Implementation Details

**Weight Tying:** Input embeddings and output head share weights, reducing params by ~38M and doubling learning efficiency.

**Scaled Initialization:** Projection layers use $\sigma = 0.02 / \sqrt{2 \times n_{layer}}$ to prevent activation explosion in deep networks.

## Resources

- [Karpathy's GPT-2 Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- GPT-2 Paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT
