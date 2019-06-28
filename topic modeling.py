#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymorphy2
import re
ma = pymorphy2.MorphAnalyzer()

def clean_text(text):
    text = text.replace("\n", " ").replace(u"╚", " ").replace(u"╩", " ").replace("sup",'')
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols 
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)
    text = text.split()#new
    new_text=[]
    for t in text:
        if t not in stopWords:
            if t!='далее' and t!='самый' and t!='свой':
                new_text.append(t)
    return new_text


# In[3]:


from stop_words import get_stop_words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

sw_list = get_stop_words('russian')+get_stop_words('english')
print(len(sw_list))
stopWords = set(stopwords.words('russian')+stopwords.words('english'))
list(stopWords)

stopWords.update(set(sw_list))
len(stopWords)


# In[4]:


import os
directory = r'C:\Users\Annie\Documents\Working\Old papers\Бюллетень оппозиции\\'
files = os.listdir(directory)

#corpus_years = {1929:[], 1930:[], 1931:[], 1932:[], 1933:[], 1934:[], 1935:[], \
           #1936:[], 1937:[], 1938:[], 1939:[], 1940:[], 1941:[]}
corpus = []
index=0

for f in files:
    if index%10==0:
        print(str(index)+" "+str(len(files)))
    index+=1
    with open(directory+f, "r") as file:
        text = clean_text(file.read())
        corpus.append(text)   


# In[7]:


write_params(corpus)


# In[24]:


corpus=restore_params()


# In[25]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[corpus], threshold=100)  


# In[11]:


# See trigram example
print(trigram[bigram[corpus[0]]])


# In[12]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[36]:


# Form Bigrams
corpus = make_trigrams(corpus)
print(corpus[1])


# In[6]:


import pickle
def write_params(corpus):
    with open('bul.dat', "wb") as file:
        pickle.dump(corpus, file)
        
def restore_params():
    with open('bul.dat', "rb") as file:
        corpus = pickle.load(file)
        return corpus


# In[9]:


import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[27]:


#jupyter notebook --NotebookApp.iopub_data_rate_limit=2147483647
# Create Dictionary
id2word = corpora.Dictionary(corpus)

# Create Corpus
texts = corpus

# Term Document Frequency
corpus_new = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus_new)


# In[28]:


tokens = 0
for t in corpus_new:
    tokens+=len(t)
print(tokens)


# In[29]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_new,
                                           id2word=id2word,
                                           num_topics=7, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[54]:


#seqmodel
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger


# In[60]:


time_slice=[50, 145, 116, 81, 76, 19, 43, 64, 57, 65, 55, 16, 7]
#time_slice=[350]
ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus_new, id2word=id2word, time_slice=time_slice, num_topics=20)


# In[61]:


with open('ldaseq20.dat', "wb") as file:
        pickle.dump(ldaseq, file)


# In[62]:


ldaseq.print_topics(time=0)


# In[30]:


# Print the Keyword in the topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus_new]


# In[31]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus_new))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[32]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherencemodel.get_coherence()
        coherence_values.append(coherence_lda)
        print('\nCoherence Score: ', coherence_lda)
        

    return model_list, coherence_values


# In[37]:


# Can take a long time to run.
limit=40
start=20
step=3
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus_new, texts=corpus, start=20, limit=40, step=3)


# In[47]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.legend(['A simple line'])
plt.show()


# In[49]:


optimal_model = model_list[4]
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))


# In[111]:


import pandas as pd
def format_topics_sentences(ldamodel=lda_model, corpus=corpus_new, texts=corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        #row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# In[114]:


def clean_text_string(text):
    text = text.replace("\n", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols 
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)
    return text

def get_year(line):
    try:
        line = line[line.index("No_")+3:line.index("_BO")]
    except:
        line = line[line.index("No_")+3:line.index(".txt")]
    if '-' in line:
        line = line[:line.index('-')]
    issue = int(line)
    if issue<=7:
        return 1929
    elif issue<=18:
        return 1930
    elif issue<=26:
        return 1931
    elif issue<=32:
        return 1932
    elif issue<=37:
        return 1933
    elif issue<=40:
        return 1934
    elif issue<=46:
        return 1935
    elif issue<=53:
        return 1936
    elif issue<=61:
        return 1937
    elif issue<=72:
        return 1938
    elif issue<=80:
        return 1939
    elif issue<=84:
        return 1940
    else:
        return 1941

corpus_years = {1929:[], 1930:[], 1931:[], 1932:[], 1933:[], 1934:[], 1935:[],            1936:[], 1937:[], 1938:[], 1939:[], 1940:[], 1941:[]}
index=0

for f in files:
    if index%10==0:
        print(str(index)+" "+str(len(files)))
    index+=1
    with open(directory+f, "r") as file:
        text = clean_text_string(file.read())
        key = int(get_year(f))
        corpus_years[key].append(text)   


# In[ ]:


#df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus_new, texts=corpus_years[1929])
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus_new, texts=corpus)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


# In[116]:


print(len(df_dominant_topic.index))#count of rows
counts = df_dominant_topic['Dominant_Topic'].value_counts().to_dict()
print(counts)


# In[ ]:


#topic distribution across all documents
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

