# BareGPT: A minimalist Transformer in NumPy

BareGPT is a Generative Pretrained Transformer developed for educational purposes, inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and the foundations of [GPT-1 (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

This project deliberately avoids deep learning frameworks: everything from the forward pass to the manual backpropagation is implemented from scratch in **less than 500 lines of NumPy**.

Because it does not rely on specific frameworks or specialized hardware, the model is ready to use immediately. A pre-trained weights file is included for instant testing, and a full training run can be completed on a standard CPU in under 30 minutes.

## Quick Start

The only dependency is NumPy:

```bash
pip install numpy
```

To run the model:

```bash
python3 train.py
```

If `data/weights.pkl` is detected, the model will load the pre-trained weights and start streaming text according to patterns learned from the training material. If no weights are found, it will begin training on the provided corpus then proceed to inference mode.

The model is trained by default on a 1MB corpus of Shakespeare’s works, although this can be replaced with any text file. The included pre-trained weights were obtained after 1,500 training steps, reaching a final loss of 1.45 and yielding coherent text generation.

Hyperparameters are available at the beginning of `train.py` to experiment with.

## How it works

BareGPT uses a Decoder-only transformer architecture, learning to predict the next character by attending to the past through a Causal Self-Attention mechanism.

Mathematically, the model generates three vectors for every token: a Query (what it is looking for), a Key (what it contains), and a Value (the actual information). By computing the dot product between Queries and Keys, the model determines an attention score, which dictates how much weight to give to each past token's Value when predicting the next one.

The training engine includes a manual implementation of the Adam optimizer with global gradient clipping for stability. During inference, the model streams its output character by character using Top-K sampling and temperature scaling to ensure fluid and varied text generation.

BareGPT is a **character-level** language model. Unlike industrial LLMs that use complex tokenizers (BPE), it treats every single character as a token. This makes the internal logic much easier to follow as the vocabulary directly maps to the alphabet and symbols found in the text.

## Sample output

A typical output with training looks like this:

```
Model configuration:
- d_model:           256
- n_layers:            2
- n_heads:             4
- batch_size:         16
- block_size:        128
Total params:  1,643,585

Training...
step 1, loss=5.447206 (0.46s)
step 100, loss=2.565365 (43.44s)
step 200, loss=2.571308 (44.86s)
step 300, loss=2.419407 (44.97s)
step 400, loss=2.320103 (45.34s)
step 500, loss=2.125059 (45.04s)
step 600, loss=2.055819 (45.32s)
step 700, loss=2.009935 (45.21s)
Weights saved to weights.pkl

Output:

The danger of and marketh friends of the cannon.

CLAUDIO:
My for the senserve your loved to his done was the
Doth even your appoises far: that well sure;
I am no not he shall be straight be thee
a regetter the damned by the bloody so love at the is,
And you at companted of did my heart. Treaf'st the danger
To treelf the forth, when I should by man is thee
That in the head it obedies contrad of your deep
To alled have welly serve to thee far to give you should but the way?

GLOUCESTER:
A miscentio, thy man, gentleward, thou can slend your
get with and enterced thou side thee be thou wilt please the king.

Lord:
I say, sir, and let us.

AUFIDIUS:
Then thy sorrow may of a death, at peoples
to the most be menation of your trad in face,
Bildren thy lay in the gentle all the death;
For my good, see world that I had prince.

KING RICHARD III:
Thyself swork, and your hand-fortail so ungracious all:
Fasticial is I had seek to children's hands.

DUKE OF YORK:
I will not faither now to be them all thy gave?
```

After training, the model, despite being character-level, successfully learns to spell words, manage indentation, and respect the structure of a theatrical play.
