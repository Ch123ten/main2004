import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

text='Real madrid is set to win the UCL for the season . Benzema might win Balon dor . Salah might be the runner up'

tokens_sents = nltk.sent_tokenize(text)
print(tokens_sents)

tokens_words = nltk.word_tokenize(text)
print(tokens_words)

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer

stem=[]
for i in tokens_words:
  ps = PorterStemmer()
  stem_word= ps.stem(i)
  stem.append(stem_word)
print(stem)

import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in stem])
print(lemmatized_output)

leme=[]
for i in stem:
  lemetized_word=lemmatizer.lemmatize(i)
  leme.append(lemetized_word)
print(leme)

print("Parts of Speech: ",nltk.pos_tag(leme))

sw_nltk = stopwords.words('english')
print(sw_nltk)

words = [word for word in text.split() if word.lower() not in sw_nltk]
new_text = " ".join(words)
print(new_text)
