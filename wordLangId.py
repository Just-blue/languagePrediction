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
parser.add_argument('--outputPath', default='wordLangId.out', help='Output path for predictions')
parser.add_argument('--unkThreshold', default=0, help='Frequency threshold to be included in vocabulary')

def wordModel(corpus, threshold, language):
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
    logger.info('Creating {0} character model'.format(language))

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
    # Calculate bigram probabilities
    mle = {bigram: (bigramFreq[bigram] + 1) / (unigramFreq[bigram[0]] + len(unigramFreq.keys())) for bigram in bigramFreq.keys()}
    return mle, unigramFreq.keys()

def main(args):
    # Load corpora and solution
    englishCorpus = readCorpus(args.englishPath)
    frenchCorpus = readCorpus(args.frenchPath)
    italianCorpus = readCorpus(args.italianPath)
    testCorpus = readCorpus(args.testPath)
    solution = loadSolution(args.solutionPath)

    # Create character models
    englishCharModel, englishVocab = wordModel(englishCorpus, args.unkThreshold, "English")
    frenchCharModel, frenchVocab = wordModel(frenchCorpus, args.unkThreshold, "French")
    italianCharModel, italianVocab = wordModel(italianCorpus, args.unkThreshold, "Italian")

    # Predict language
    charResults = predictLanguage(testCorpus, englishCharModel, englishVocab, frenchCharModel, frenchVocab, italianCharModel, italianVocab, wordModel=True)
    evaluate(charResults, solution)
    writeResults(charResults, args.outputPath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)