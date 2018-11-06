"""
Author: Lauren Gardiner
Date: 11/1/18
"""
import string
import operator
from collections import Counter
import math
import logging

def readCorpus(filepath):
    """ This function takes a file path and returns the cleaned corpus as a list of lines

    It cleans the corpus by removing punctuation and reducing all whitespace to a single space. It utilizes utf-8 encoding to work with all languages and surrogate escaping to handle special characters.
    
    Args:
        filepath (str): filepath to the corpus text file
    
    Returns:
        lines (list): cleaned corpus as a list of lines
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading {0} corpus'.format(filepath))

    with open(filepath, 'r', encoding='utf-8', errors="surrogateescape") as f:
        corpus = f.read()
    for p in string.punctuation:
        corpus = corpus.replace(p, "")
    corpus = corpus.replace(r"\s", " ")
    lines = corpus.lower().splitlines()
    return lines

def loadSolution(filepath):
    """ This function takes a file path and returns a list of the correct language without the index

    Args:
        filepath (str): filepath to the soution file

    Returns:
        solution (list): list of labels for the test corpus
    """ 
    logger = logging.getLogger(__name__)
    logger.info('Loading solution file')

    with open(filepath, 'r') as f:
        lines = f.readlines()
    solution = [line.split()[1] for line in lines]
    return solution

def createOOV(unigramFreq, threshold):
    """ This function takes a frequency dictionary and OOV threshold and returns an updated dictionary and OOV list

    If a character/word is removed from the vocabulary, it's frequency is shifted to the key <unk>.

    Args:
        unigramFreq (dict): a dictionary of unigram keys and frequency values
        threshold (int): frequency value a character/word must be seen more than to be included in the vocabulary

    Returns:
        unigramFreq (dict): a updated dictionary of unigram keys and frequency values
        OOV (list): a list of word removed from the vocabulary based on the frequency threshold
    """
    logger = logging.getLogger(__name__)
    logger.info('Removing out of vocabulary characters/words')
    OOV = []
    unkCount = 0
    for c, freq in unigramFreq.items():
        if freq < int(threshold):
            OOV.append(c)
            unkCount += freq
    unigramFreq['<unk>'] = unkCount
    for c in OOV:
        del unigramFreq[c]
    logger.info('{0} were removed from the vocabulary'.format(len(OOV)))
    return unigramFreq, OOV

def calcLangProbs(line, model, vocab):
    """ Given a line from a corpus, a language model, and vocabulary, return the probability of the line being from the language

    Args:
        line (list): list of characters/words from a line in a corpus
        model (dict): a dictionary of bigram keys and probability values for all seen bigrams
        vocabulary (list): vocabulary for the language corpus
    
    Returns:
        langProb (float): probability that the line belongs to that language
    """
    langProb = 0
    for i in range(len(line) - 1):
        char1 = line[i]
        char2 = line[i+1]
        if char1 not in vocab:
            char1 = '<unk>'
        if char2 not in vocab:
            char2 = '<unk>'
        bigram = (char1, char2)
        if (model.get(bigram, 0) == 0):
            bigram = ('<unk>', '<unk>')
        langProb += math.log(model[bigram])
    return langProb

def predictLanguage(testCorpus, englishCharModel, englishVocab, frenchCharModel, frenchVocab, italianCharModel, italianVocab, wordModel):
    """ Given the test corpus, language models, and vocabulary return the language prediction for each line in the test corpus

    Args:
        testCorpus (list): list of lines from the test corpus
        englishCharModel (dict): a dictionary of bigram keys and probability values for all seen bigrams in the English corrpus
        englishVocab (list): vocabulary for the English corpus
        frenchCharModel (dict): a dictionary of bigram keys and probability values for all seen bigrams in the French corrpus
        frenchVocab (list): vocabulary for the French corpus
        italianCharModel (dict): a dictionary of bigram keys and probability values for all seen bigrams in the Italian corrpus
        italianVocab (list): vocabulary for the Italian corpus

    Returns:
        results (list): prediction for each line in the test corpus
    """
    logger = logging.getLogger(__name__)
    logger.info('Predicting languages for {0} lines in the test corpus'.format(len(testCorpus)))

    results = []
    for line in testCorpus:
        # Add start and end tokens
        if wordModel:
            line = '<start> ' + line + ' <end>'
            line = line.split()
        else:
            line = ['<start>'] + list(line) + ['<end>']
        predictions = {}
        predictions['English'] = calcLangProbs(line, englishCharModel, englishVocab)
        predictions['French'] = calcLangProbs(line, frenchCharModel, frenchVocab)
        predictions['Italian'] = calcLangProbs(line, italianCharModel, italianVocab)
        # Select the language with the highest probability
        results.append(max(predictions.items(), key=operator.itemgetter(1))[0])
    return results

def evaluate(results, solution):
    """ Given a list of results and their ground truth labels, calculate the accuracy and log incorrect predictions

    Args:
        results (list): list of language predictions
        solution (list): ground truth labels
    """
    logger = logging.getLogger(__name__)
    logger.info('Evaluating results')

    accuracy = sum(1 for r,s in zip(results,solution) if r == s) / float(len(solution))

    logger.info('Accuracy is {0}'.format(accuracy))

    wrong = 0
    for i, lang in enumerate(zip(results,solution)):
        if lang[0] != lang[1]:
            logger.info("Line {0} is wrong. You predicted {1}, but it's actually {2}".format(i + 1, lang[0], lang[1]))
            wrong += 1
    logger.info("{0} were predicted incorrectly".format(wrong))

def writeResults(results, filepath):
    """ Given a list of predictions, write out the results to the output path

    Args:
        results (list): list of language predictions
        filepath (str): filepath to save results
    """
    logger = logging.getLogger(__name__)
    logger.info('Writing results to {0}'.format(filepath))
    with open(filepath, "w") as f:
        for i, line in enumerate(results):
            f.write(str(i + 1) + "\t" + line + "\n")