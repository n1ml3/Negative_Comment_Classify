import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tự động download các gói cần thiết nếu chưa có
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop = stopwords.words('english')
wn = WordNetLemmatizer()
my_sw = ['rt', 'ht', 'fb', 'amp', 'gt']

def decontracted(st):
    st = re.sub(r"won\'t", "will not", st)
    st = re.sub(r"can\'t", "can not", st)
    st = re.sub(r"n\'t", " not", st)
    st = re.sub(r"\'re", " are", st)
    st = re.sub(r"\'s", " is", st)
    st = re.sub(r"\'d", " would", st)
    st = re.sub(r"\'ll", " will", st)
    st = re.sub(r"\'ve", " have", st)
    st = re.sub(r"\'m", " am", st)
    return st

def clear_link(st):
    word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?%\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?%\:@\-_=#])*', 
                '', st, flags=re.MULTILINE)
    word = re.sub(r'(@[^\s]*)', "", word)
    return word

def clear_punctuation(st):
    word = re.sub(r'[^\w\s]', '',st)
    return word

def clear_special(st):
    word = re.sub('[^a-zA-Z]', ' ', st)
    return word

def clear_noise(word):
    word = word.lower()
    word = decontracted(word)
    word = clear_link(word)
    word = clear_punctuation(word)
    word = clear_special(word)
    return word

def clear_stopwords(st):
    word = " ".join(st for st in st.split() if st not in stop)
    return word

def black_txt(token):
    if token == 'u':
        token = 'you'
    return  token not in stop and token not in list(string.punctuation) and token not in my_sw

def fun_stemlem(word):
    list_word_clean = []
    for w1 in word.split(" "):
        if  black_txt(w1.lower()):
            word_lemma =  wn.lemmatize(w1,  pos="v")
            list_word_clean.append(word_lemma)
    word = " ".join(list_word_clean)
    return word 

def prepare_data(word):
    if not isinstance(word, str):
        return ""
    word = clear_noise(word)
    word = clear_stopwords(word)
    word = fun_stemlem(word)
    return word
