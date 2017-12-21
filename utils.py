from collections import Counter
from nltk.corpus import stopwords, words
from nltk.stem.snowball import SnowballStemmer
import json, pandas, re


#stemming_____________________________________________________________________________________________________________
stemmer = SnowballStemmer('english')


#contraction__________________________________________________________________________________________________________
contraction = json.load(open('constants/contraction.json'))


#stopword_____________________________________________________________________________________________________________
stop_words = stopwords.words('english')
stop_words.remove('not')



#correction__________________________________________________________________________________________________________
def all_words(text):
    return re.findall('\\w+', text.lower())

WORDS = Counter(all_words(open('constants/big.txt').read()))

def P(word, N=sum(WORDS.values())):
    """Probability of `word`."""
    return WORDS[word] / N

def correction(word):
    """Most probable spelling correction for word."""
    return max(candidates(word), key=P)

def candidates(word):
    """Generate possible spelling corrections for word."""
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

def known(words):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set((w for w in words if w in WORDS))

def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [ (word[:i], word[i:]) for i in range(len(word) + 1) ]
    deletes = [ L + R[1:] for L, R in splits if R ]
    transposes = [ L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1 ]
    replaces = [ L + c + R[1:] for L, R in splits if R for c in letters ]
    inserts = [ L + c + R for L, R in splits for c in letters ]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


#word dictionary______________________________________________________________________________________________________
word_dictionary = list(set(words.words()))
for alphabet in 'bcdefghjklmnopqrstuvwxyz':
    word_dictionary.remove(alphabet)

useless_two_letter_words = pandas.read_csv('constants/useless_two_letter_words.csv')
for word in useless_two_letter_words:
    word_dictionary.remove(word)

useful_words = pandas.read_csv('constants/useful_words.csv')
for word in useful_words:
    word_dictionary.append(word)

for key in contraction:
    word_dictionary.append(key)


#split hashtags_______________________________________________________________________________________________________
def split_hashtag(hashtag):
    found = False
    for i in reversed(range(1, len(hashtag) + 1)):
        if stemmer.stem(hashtag[:i]) in word_dictionary or hashtag[:i] in word_dictionary:
            found = True
            if i == len(hashtag):
                if hashtag[:i] in contraction:
                    contraction[hashtag[:i]]
                else:
                    return hashtag[:i]
            else:
                child = split_hashtag(hashtag[i:])
                if child:
                    return hashtag[:i] + ' ' + child
    if not found:
        return False
