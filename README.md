# Text Analytics Homework 2
## Lauren Gardiner

## Overview & Design

This project predicts whether a line from a test corpus is one of three languages: English, French, or Italian. 

All related text files (training corpora, test corpus, and solution file) should be in the same directory. If the are located in another location, please specify with `--englishPath`, `--frenchPath`, `--italianPath`, `--testPath`, and `--solutionPath`. The character model runs with the defualt unknown threshold of 30, indicating that all characters must be seen 30 times in the training corpus to be included in a language's vocabulary. The word models run with the default unknown threshold of 0, indicating that all words from the training corpora will be included in the vocabulary. To change this, utilize the `--unkThreshold` argument.

All lines in the corpora were cleaned by removing punctuation, casing, and extra whitespace to focus the lanugage model on the relevant text. Additionally, `<start>` and `<end>` tokens were added to each line so the probability of the beginning and end of a line in combination with any other gram is captured.

## Getting Started

To run the character language model with add smoothing and the default arguments:

```
python letterLangId.py
```

To run the word language model with add smoothing and the defualt arguments:

```
python wordLangId.py
```

To run the word language model with Good Turing smoothing and the defualt arguments:

```
python wordLangId2.py
```

## Questions & Performance Analysis
### 1)
The letter bigram model cannot be implemented without smoothing because unknown bigrams would results in 0 frequency values leading to issues when calculating the entire sentence's conditional probability. Also, if the entire training set is included in the vocabulary, a probably can't be calculated for bigrams that begin with an unknown word due to errors when dividing by 0. This problem can be resolved with add one smoothing because it removes 0 counts in the data. The letter bigram model with add one smoothing **correctly predicted 297/300** of the lines in the test corpus. Given the strong performance with add one smoothing, it seems like an effective solution to the zero count problem.

In addition to add one smoothing, I set the unknown threshold to 30. Given the overall frequency of a character within our corpora, a threshold of 30 will not affect that vocabulary size, but just remove erroneous characters. However, this threshold of 30 (which can be changed by the user) does not seem to affect performance and yields the same results as a threshold of 0. 

### 2)
Just like the letter bigram model, the word bigram model cannot be implemented without smoothing either unless you set a threshold for the training set to be included in the vocabulary and introduce unknown tokens. Since the corpora is small and in turn, the vocabulary, it's best to implement smoothing versus a threshold for out of vocabulary words. Therefore, the default threshold is set to 0 to not drastically reduce the vocabulary. If the training corpora was exponentially larger, the user may want to consider a threshold of 30 to manage the vocabulary size and reduce the risk of including typos from a corpus.

Smoothing is even more important for a word level model than character level due to the larger number of words than characters in a given language and in turn, a higher occurence of rare events. Given the simplicity of add one smoothing versus other smoothing techniques, I implemented it and it results in **predictions that were correct 298/300 times**. 

### 3)
Another approach for handling zero frequencies than add one smoothing that I evaluated is Good Turing smoothing. This method adjusts the frequency to shift some of the probability to unseen events. Rather than blindly adding one to each frequency, it takes into account the existing frequencies in the training corpus. I applied the same unknown threshold of 0 for the reasons described in question two and saw results of **299/300 correct predictions**.

Given the results from my experiement, **the language model with Good Turing smoothing performed the best** and therefore, should be utilized. Given the small size of our corpora, computational efficiency is not a huge consideration and therefore, performance should be prioritized. If our corpora was much larger and/or training time was a consideration, the word model with add one smoothing may need to be considered due to the computational cost of adjusting frequencies. If memory constraints were an issue, a character level model would have a smaller vocabulary and could potentially perform better than a word level model with a heavily pruned vocabulary. However, given the tradeoffs and the use case of this homework, I would go with the language model with Good Turing smoothing and an unknown threshold of 0.