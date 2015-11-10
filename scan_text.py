import matplotlib.pyplot as pyplot

import nltk
import string
from nltk.corpus import stopwords
stopset = set(stopwords.words('russian'))
stopset = stopset.union({'это','весь','свой','который'})
stopset_light = {'а','и','но','как','что','который',
                 'в','на','о','по','с','у','к','за','из','от','под',
                 'не','ни','да','бы','ли','же','как','так','то',
                 'я','мы','ты','вы','он','она','оно','они',
                 'мой','наш','твой','ваш','его','ее','их','свой',
                 'меня','нас','тебя','вас','себя',
                 'быть','весь','это','этот','тот',''
                 }

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

import pymorphy2

def make_russian_plots():
    '''change font to russian-friendly'''
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Verdana'
    return

### ---work with words--- ###
   
def tokenize(text):
    '''text tokenizator'''
    
    # tokenization
    words = nltk.word_tokenize(text)
    
    # delete punctuation
    punctuation = string.punctuation + '–'+'—'+'…'+'...'+'..'
    words = [w for w in words if (w not in punctuation)]
    words = [w for w in words if not w.isdigit()]
    
    # cleaning words
    words = [w.replace('\'','').replace('`','') for w in words]
    words = [w.replace('«','').replace('»','') for w in words]
    words = [w.replace('„','').replace('“','') for w in words]
    words = [w.replace('…','') for w in words]
    words = [w.replace('—','') for w in words]
    words = [w.replace('\ufeff','') for w in words]
    
    # make letters small
    words = [w.lower() for w in words if w != '']
    
    # make NLTK text
    text = nltk.text.Text(words)
    
    return text
    
def normalize(text):
    '''makes morphological analisis'''
    morph = pymorphy2.MorphAnalyzer()
    words = [morph.parse(w)[0].normal_form for w in text]
    return nltk.Text(words)
    
    
### ---work with texts--- ###

def filter_stops(word):
    '''filter for stop words set'''
    return word in stopset
    
def filter_stops_light(word):
    '''filter for light stop words set'''
    return word in stopset_light
    
def apply_filter(text, filter_to_use):
    '''filters text with filter_to_use'''
    return nltk.Text([w for w in text if not filter_to_use(w)])
    
def lexical_diversity(text):
    '''lexical diversity'''
    return len(set(text)) / len(text)
    
def find_bigrams(text, n, freq=10):
    ''''find n most popular bigrams'''
    bcf = BigramCollocationFinder.from_words(text)
    bcf.apply_freq_filter(freq)
    return bcf.nbest(BigramAssocMeasures.likelihood_ratio, n)
    
def find_trigrams(text, n, freq=10):
    ''''find n most popular trigrams'''
    tcf = TrigramCollocationFinder.from_words(text)
    tcf.apply_freq_filter(freq)
    return tcf.nbest(TrigramAssocMeasures.likelihood_ratio, n)
    
def words_with_length(text, length):
    '''finds word with length more then length'''
    return [w for w in text.vocab() if len(w) == length]
    
def words_length(text):
    '''word length frequency distribution'''
    return nltk.FreqDist([len(w) for w in text.vocab()])
    
### ---plots--- ###
    
def plot_most_common(text, n):
    '''plot dispersion for n most common words'''
    voc = text.vocab()
    fd = voc.most_common(n)
    voc_most_common = [i[0] for i in fd]
    print(voc_most_common)
    text.dispersion_plot(voc_most_common)
    return
    
def plot_words_length(text):
    '''plot words length'''
    wl = words_length(text)
    lengths = [i for i in wl]
    freqs = [wl.freq(i) for i in wl]
    
    
#    pyplot.figure(figsize=(10,6))
    pyplot.plot(lengths, freqs)
    pyplot.title("Word Length Frequency")
    pyplot.xlabel("Word Length")
    pyplot.ylabel("Frequency")
    pyplot.show()
    
    return

### ---io--- ###

def read_file(file):
    '''read text file'''  
    
    print(file)
    try:
        f = open(file, 'r')
        text = f.read()
    except:
        f = open(file, encoding='UTF8')
        text = f.read()
    f.close()
    
    # tokenize text
    text = tokenize(text)
    
    return text
    
###### ---usage--- ######
    
def analize(file):
    '''analize text file'''  
    
    # read file
    text_init = read_file(file)
    text_filter = apply_filter(text_init, filter_stops)
    text = normalize(text_init)
    text = apply_filter(text, filter_stops)
    
    # statistics
    print('\n')
    print('text length       = {0} words'.format(len(text_init)))
    print('vocabulary length = {0} words'.format(len(text.vocab())))
    print('lexical dive1rsity = {0}\n'.format(lexical_diversity(text)))
    
    # collocations
    print('bigrams: {0}\n'.format(find_bigrams(text_filter, 10)))
#    print('trigrams: {0}\n'.format(find_trigrams(text_filter, 10)))
    
    # plots
    plot_words_length(text)
    plot_most_common(text, 10)
    return
    
    