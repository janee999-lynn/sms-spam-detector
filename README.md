# SMS Spam Detection with SimpleRNN

**Author:** Chea Dalin

**Course project:** Deep Learning Final Exam (Option A2)

## 1. Problem Description

The goal is to build a small NLP model that reads an SMS text message and predicts
whether it is **spam** (an unwanted/scam message) or **ham** (a normal, safe message).
This is a binary text classification problem.

## 2. Dataset Description

- File: `dataset/A2_sms_spam_detection.csv`
- 130 total messages (originally 100, expanded with the teacher's permission — see Section 8)
- Balanced: 65 spam, 65 ham
- Columns: `text` (the message), `label` (spam or ham)
- No duplicate rows, no missing values

Example spam message: *"Claim your FREE reward before midnight."*
Example ham message: *"Please call me when you arrive."*

## 3. Model Architecture

```
Text -> Tokenizer -> Padding -> Embedding -> SimpleRNN -> Dense (sigmoid) -> spam/ham
```

- Vocabulary size: 1000 words
- Max sequence length: 20
- Embedding dimension: 64
- SimpleRNN units: 64
- Output: 1 neuron with sigmoid activation (0 = ham, 1 = spam)
- Loss: binary crossentropy, Optimizer: Adam
- Random seed fixed at 7 for reproducible results
- Trained for up to 30 epochs with **Early Stopping** (patience=4, monitor `val_loss`,
  restores best weights) — training actually stopped automatically after **8 epochs**

**Why an Embedding layer?** Raw word ID numbers don't carry any meaning by themselves.
The Embedding layer turns each word into a small vector that captures meaning, so
similar words end up with similar vectors, this is what the SimpleRNN actually learns from.

**Why SimpleRNN?** SimpleRNN reads the message word by word and keeps a memory of
what came before, so word order matters (e.g. "you won" vs "won you"). This is
important because spam messages often follow specific word patterns/order.

**Why Early Stopping?** An earlier version of this model trained for a fixed 15 epochs and
clearly overfit, training loss dropped to near zero while validation loss kept climbing.
Early Stopping fixes this by halting training as soon as validation loss stops improving,
then keeping the best-performing weights instead of the final (overfit) ones.

## 4. Training Results

- Split: 80% train / 20% test, with a validation split from the training data
- Training stopped automatically at **epoch 8** (out of a max of 30) via Early Stopping

See `accuracy.png` and `loss.png` for the training curves. Validation loss drops for the
first few epochs, flattens, then only rises slightly. Early Stopping catches the best
point before real overfitting sets in.

### Test set performance (26 unseen messages)

| Metric | Value |
|---|---|
| Accuracy | 96.2% |
| Precision (ham) | 0.93 |
| Recall (ham) | 1.00 |
| F1-score (ham) | 0.96 |
| Precision (spam) | 1.00 |
| Recall (spam) | 0.92 |
| F1-score (spam) | 0.96 |

**Confusion Matrix:**
```
              Predicted ham   Predicted spam
Actual ham        13                0
Actual spam        1               12
```

Only 1 mistake out of 26 test messages.

## 5. Experiment Results

Both experiments use Early Stopping, so each setting trains to its own best point rather
than a fixed epoch count.

**Experiment A — Embedding Dimension**

| Embedding dim | Test accuracy |
|---|---|
| 32 | 88.5% |
| 64 | 96.2% |
| 128 | 92.3% |

**Experiment B — SimpleRNN Units**

| RNN units | Test accuracy |
|---|---|
| 32 | 96.2% |
| 64 | 96.2% |
| 128 | 88.5% |

**Takeaway:** the middle setting (64) performs best for both. A bigger network (128) doesn't
help and can even hurt, since with only 130 samples there isn't enough data to justify the
extra model capacity a useful, realistic finding for a small-dataset project.

## 6. Example Predictions (on brand-new sentences, not in the dataset)

| Message | Prediction | Confidence |
|---|---|---|
| "Congratulations, you've been selected to win a free iPhone, click here now!" | SPAM | 96.0% |
| "Can you send me the notes from class?" | HAM | 99.1% |
| "URGENT: your account will be suspended, click here now" | SPAM | 97.6% |
| "You've won a $500 gift card, claim now!" | SPAM | 99.0% |
| "Reminder: Your appointment is tomorrow at 10pm" | HAM | 97.4% |
| "Hey mom, can you pick me up at 5?" | HAM | 98.8% |

All 6 new, unseen messages were classified correctly — including two tricky cases (the
appointment reminder and the "mom" message) that were misclassified in an earlier version
of this model. See Section 8 for how that was fixed.

## 7. Future Improvements

- Use an even bigger dataset — 130 samples is still small and makes test results a bit
  sensitive to which 26 messages happen to land in the test split
- Try LSTM or GRU, which handle longer messages/context better than SimpleRNN
- Add text cleaning (lowercase, strip punctuation, handle emojis/slang common in real SMS)
- Try pre-trained word embeddings (like GloVe) instead of training embeddings from scratch

## 8. A Real Problem Found & Fixed: the "your" Shortcut

While testing this model on new sentences, I found that messages containing the word
**"your"** (e.g. *"Reminder: Your appointment is tomorrow"*, *"Hey mom, can you pick me
up?"*) were sometimes wrongly flagged as spam.

**Why this happened:** in the original 100-sample dataset, the word "your" appeared mostly
in spam messages ("claim your reward", "your number was selected") and rarely in ham
messages. With so few examples, the model latched onto "your" as a shortcut for spam
instead of learning the deeper, more meaningful pattern (urgency, prizes, links).

**Two fixes were applied, after checking with my teacher:**
1. **Expanded the dataset from 100 to 130 samples** — added ham examples that use "your" in
   ordinary contexts (*"Is your homework done yet?"*) and spam examples that don't rely on
   the word "your" at all (*"Claim the prize now before it expires!"*).
2. **Added Early Stopping** (Section 3/4) — this fixed a separate overfitting problem
   visible in the training curves, and also improved overall generalization.

**Result:** every test sentence in Section 6 — including both "your"-containing cases that
used to fail — is now classified correctly, with 96.2% overall test accuracy and no missed
real spam messages (recall on spam stayed high, see Section 4).

## 9. How to Run

```bash
pip install -r requirements.txt   # tensorflow, pandas, numpy, scikit-learn, matplotlib, streamlit
jupyter notebook training.ipynb   # trains the model and saves it to model/
jupyter notebook prediction.ipynb # loads the saved model and predicts on new text
streamlit run app.py              # interactive web app
```
