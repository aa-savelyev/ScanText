import matplotlib.pyplot as pyplot

import nltk
import string
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

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

ru_POS = {'NOUN':'Cуществительные','INFN':'Глаголы (инфинитивы)','ADJF':'Прилагательные',
          'ADVB':'Наречия','NUMR':'Числительные','NPRO':'Местоимения','PREP':'Предлоги',
          'CONJ':'Союзы','PRCL':'Частицы','INTJ':'Междометия',
          'ADJS':'Краткие прилагательные','COMP':'Компаративы','VERB':'Глаголы (личная форма)',
          'PRTF':'Причастия','PRTS':'Краткие причастия','GRND':'Деепричастия',
          'PRED':'Предикативы',None:'Неопределено'}

def make_russian_plots():
    '''change font to russian-friendly'''
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Verdana'
    return

### ---work with words--- ###
   
def word_tokenize(text):
    '''text word tokenizator'''  
    # word tokenization
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

def sent_tokenize(text):
    '''text sentence tokenizator'''
    # sentence tokenization
    sent = nltk.sent_tokenize(text, language = 'russian')
    return sent
    
def normalize(words):
    '''makes morphological analysis'''
    parts = [morph.parse(w)[0] for w in words]
    words = [p.normal_form for p in parts if p.tag.POS != None]
    POS = nltk.FreqDist([p.tag.POS for p in parts if p.tag.POS != None])
    return nltk.Text(words), POS
    
### ---work with texts--- ###

def filter_stops(word):
    '''filter for stop words set'''
    return word in stopset
    
def filter_stops_light(word):
    '''filter for light stop words set'''
    return word in stopset_light
    
def apply_filter(words, filter_to_use):
    '''filters text with filter_to_use'''
    return nltk.Text([w for w in words if not filter_to_use(w)])
    
def lexical_diversity(words):
    '''lexical diversity'''
    words_norm = normalize(words)
    vocab = words_norm.vocab()
    return len(words) / len(vocab)
    
def find_bigrams(words, n, freq=10):
    ''''find n most popular bigrams'''
    bcf = BigramCollocationFinder.from_words(words)
    bcf.apply_freq_filter(freq)
    return bcf.nbest(BigramAssocMeasures.likelihood_ratio, n)
    
def find_trigrams(words, n, freq=10):
    ''''find n most popular trigrams'''
    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_freq_filter(freq)
    return tcf.nbest(TrigramAssocMeasures.likelihood_ratio, n)
      
def tokens_length(tokens):
    '''token length frequency distribution'''
    return nltk.FreqDist([len(t) for t in tokens])
    
def tokens_with_length(tokens, length):
    '''finds token with length equal to length'''
    return [t for t in tokens if len(t) == length]
    
def get_POS(words):
    '''part of speech statistics'''
    POS = nltk.FreqDist([morph.parse(w)[0].tag.POS for w in words])
    return POS
    
### ---output--- ###
    
def plot_most_common(words, n):
    '''plot dispersion for n most common words'''
    vocab = words.vocab()
    fd = vocab.most_common(n)
    vocab_most_common = [i[0] for i in fd]
    print(vocab_most_common)
    words.dispersion_plot(vocab_most_common)
    return
    
def plot_words_length(words):
    '''plot words length'''
    wl = tokens_length(words)
    lengths = sorted([i for i in wl])
    freqs = [wl.freq(i) for i in wl]    
    
#    pyplot.figure(figsize=(10,6))
    pyplot.plot(lengths, freqs)
    pyplot.title("Word Length Frequency")
    pyplot.xlabel("Word Length")
    pyplot.ylabel("Frequency")
    pyplot.show()
    
    mean = sum([lengths[i]*freqs[i] for i in range(len(wl))])
    print("Most popular length = {0}".format(wl.most_common()[0][0]))
    print("Mean length         = {0:.2f}".format(mean))
    return
    
def plot_sents_length(sents):
    '''plot sentences length'''
    sl = tokens_length(sents)
    lengths = sorted([i for i in sl])
    freqs   = [sl.freq(i) for i in sl]    
    
#    pyplot.figure(figsize=(10,6))
    pyplot.plot(lengths, freqs)
    pyplot.title("Sentence Length Frequency")
    pyplot.xlabel("Sentence Length")
    pyplot.ylabel("Frequency")
    pyplot.show()
    
    mean = sum([lengths[i]*freqs[i] for i in range(len(sl))])
    print("Most popular length = {0}".format(sl.most_common()[0][0]))
    print("Mean length         = {0:.2f}".format(mean))
    return
    
#def POS_rus(POS):
#    '''returns russian name of the'''
#    switch (POS):
#        case
    
def print_POS(POS):
    '''print part of speech statistics'''
    S = sum([POS[i] for i in POS])
    POS_sorted = POS.most_common()
    
    for part in POS_sorted:
        print("{0:<25} {1:.2%}".format(ru_POS[part[0]], part[1]/S))
            
    return
## ---io--- ###

def read_file(file):
    '''read text file'''  
#    print(file)
    try:
        f = open(file, 'r')
        text = f.read()
    except:
        f = open(file, encoding='UTF8')
        text = f.read()
    f.close()
    
    return text
    
###### ---usage--- ######
    
def analyze(file):
    '''analyze text file'''  
    # read file
    text = read_file(file)
    sents = sent_tokenize(text)
    words = word_tokenize(text)
    words_norm, POS = normalize(words)
    
    # statistics
    vocab = words_norm.vocab()
    hapaxes = vocab.hapaxes()
    print('\n')
    print("Text length           = {0} words".format(len(words)))
    print("Vocabulary length     = {0} words".format(len(vocab)))
    print("Lexical diversity     = {0:.2}".format(len(words)/len(vocab)))
    print("Percentage of hapaxes = {0:.1%}\n".format(len(hapaxes)/len(vocab)))
    
    # parts of speech
    POS.plot()
    print_POS(POS)
    
    # collocations
    words_filter = apply_filter(words, filter_stops)
    print('\n')
    print("Bigrams: {0}\n".format(find_bigrams(words_filter, 10)))
#    print("Trigrams: {0}\n".format(find_trigrams(words_filter, 10)))
    
    # plots
    make_russian_plots()
    plot_words_length(words_filter)
    plot_sents_length(sents)
    
    words_norm_filter = apply_filter(words_norm, filter_stops)
    plot_most_common(words_norm_filter, 10)
    
    return words_norm_filter
    
    