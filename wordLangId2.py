"""
Author: Lauren Gardiner
Date: 11/1/18
"""
from collections import Counter
import argparse
import logging
from helper import readCorpus, loadSolution, createOOV, calcLangProbs, predictLanguage, evaluate, writeResults

parser = argparse.ArgumentParser(description='Word level language model')
parser.add_argument('--englishPath', default='LangId.train.English', help='Input path for English corpus')
parser.add_argument('--frenchPath', default='LangId.train.French', help='Input path for French corpus')
parser.add_argument('--italianPath', default='LangId.train.Italian', help='Input path for Italian corpus')
parser.add_argument('--testPath', default='LangId.test', help='Input path for test corpus')
parser.add_argument('--solutionPath', default='LangId.sol', help='Input path for solution')
parser.add_argument('--outputPath', default='wordLangId2.out', help='Output path for predictions')
parser.add_argument('--unkThreshold', default=0, help='Frequency threshold to be included in vocabulary')

def findGTCutoff(N_c):
    """Given a dictionary of the numbers of grams seen at each frequency, return a cutoff for good turing smoothing

    Args:
        N_c (dict): a dictionary of the numbers of grams seen at each frequency
    
    Returns:
        cutoff (int): cutoff for Good Turing smoothing
    """
    for i in range(1, len(N_c)):
        if i in N_c.keys():
            pass
        else:
            return i - 1
            
def goodTuringSmoothing(unigramFreq, bigramFreq, unkBigrams):
    """ Given a unigram frequency dictionary, a bigram frequency dictionary, and a number of unknown bigrams, return frequency dictionaries with good turing smoothing

    Args:
        unigramFreq (dict): a dictionary of unigram keys and frequency values
        bigramFreq (dict): a dictionary of bigram keys and frequency values
        unkBigram (int): number of unknown bigrams given all possible vocabulary combinations
    
    Returns:
        unigramGTFreq (dict): a dictionary of unigram keys and frequency values
        bigramGTFreq (dict): a dictionary of bigram keys and frequency values

    """
    # Inital dictionary for adjusted frequencies
    unigramGTFreq = {}
    bigramGTFreq = {}
    # Calculate the numbers of unigrams and bigrams seen at each frequency
    unigramN_c = dict(Counter(unigramFreq.values()))
    bigramN_c = dict(Counter(bigramFreq.values()))
    # Find the unigram and bigram cutoff for good turing smoothing
    unigramCutoff = findGTCutoff(unigramN_c)
    bigramCutoff = findGTCutoff(bigramN_c)
    # Adjust frequencies for unigrams
    for k, c in unigramFreq.items():
        if c < unigramCutoff:
            unigramGTFreq[k] = (c + 1) * (unigramN_c[c+1] / unigramN_c[c])
        else:
            unigramGTFreq[k] = c
    # Adjust frequencies for bigrams
    for k, c in bigramFreq.items():
        if c < bigramCutoff:
            bigramGTFreq[k] = (c + 1) * (bigramN_c[c+1] / bigramN_c[c])
        else:
            bigramGTFreq[k] = c
        if k == ("<unk>", "<unk>"):
            bigramGTFreq[k] = (bigramN_c[1] / unkBigrams)
    return unigramGTFreq, bigramGTFreq

def wordModel(corpus, threshold, language, smoothing="addOne"):
    """ This function take a corpus and an OOV threshold and returns the vocabulary and the probabilities of seeing two words.

    Words that are seen less than the threshold are converted to an unknown token <unk>

    Args:
        corpus (list): list of lines from a language corpus
        threshold (int): frequency value a word must be seen more than to be included in the vocabulary

    Returns:
        mle (dict): a dictionary of bigram keys and probability values for all seen bigrams
        unigramFreq.keys() (list): vocabulary for the language corpus
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating {0} word model'.format(language))

    # Add start and end sentence tokens in each line
    corpus = ['<start> ' + line + ' <end>' for line in corpus]
    # Create a dictionary with character keys and frequency values
    unigramFreq = sum([Counter(line.split()) for line in corpus], Counter())
    unigramFreq, OOV = createOOV(unigramFreq, threshold)
    # Add unknown tokens to dictionary if there are none due to a threshold of 0
    if unigramFreq.get('<unk>', 0) == 0:
        unigramFreq['<unk>'] = 0
    
    # Create a dictionary with bigram keys and frequency values
    bigramFreq = {}
    for line in corpus:
        line = line.split()
        for i in range(len(line) - 1):
            word1 = line[i]
            word2 = line[i+1]
            if word1 in OOV:
                word1 = '<unk>'
            if word2 in OOV:
                word2 = '<unk>'
            bigram = (word1, word2)
            if bigram in bigramFreq.keys():
                bigramFreq[bigram] += 1
            else:
                bigramFreq[bigram] = 1
    # Add unknown tokens to dictionary if there are none due to a threshold of 0
    if bigramFreq.get(('<unk>', '<unk>'), 0) == 0:
        bigramFreq[('<unk>', '<unk>')] = 0
    # Create unknowns for all unseen bigrams given the vocabulary
    unkBigrams = 0
    for unigram1 in unigramFreq.keys():
        for unigram2 in unigramFreq.keys():
            if bigramFreq.get((unigram1, unigram2), 0) == 0:
                unkBigrams += 1
    # Calculate bigram probabilities
    if smoothing == "addOne":
        mle = {bigram: (bigramFreq[bigram] + 1) / (unigramFreq[bigram[0]] + len(unigramFreq.keys())) 
               for bigram in bigramFreq.keys()}
    elif smoothing == "GT":
        unigramGTFreq,bigramGTFreq = goodTuringSmoothing(unigramFreq, bigramFreq, unkBigrams)
        mle = {bigram: (bigramGTFreq[bigram] / sum(bigramGTFreq.values())) / (unigramGTFreq[bigram[0]] / sum(unigramGTFreq.values())) for bigram in bigramGTFreq.keys()}
    return mle, unigramFreq.keys()

def main(args):
    # Load corpora and solution
    englishCorpus = readCorpus(args.englishPath)
    frenchCorpus = readCorpus(args.frenchPath)
    italianCorpus = readCorpus(args.italianPath)
    testCorpus = readCorpus(args.testPath)
    solution = loadSolution(args.solutionPath)

    # Create character models
    englishWordModel, englishVocab = wordModel(englishCorpus, args.unkThreshold, "English", "GT")
    frenchWordModel, frenchVocab = wordModel(frenchCorpus, args.unkThreshold, "French", "GT")
    italianWordModel, italianVocab = wordModel(italianCorpus, args.unkThreshold, "Italian", "GT")

    # Predict language
    wordResults = predictLanguage(testCorpus, englishWordModel, englishVocab, frenchWordModel, frenchVocab, italianWordModel, italianVocab, wordModel=True)
    evaluate(wordResults, solution)
    writeResults(wordResults, args.outputPath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)