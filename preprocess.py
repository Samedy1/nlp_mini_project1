# %% [markdown]
# ## 1. Data Preparation

# %%
# read file
# import os

# file_names = os.listdir('corpus')
# file_content = ''

# for file_name in file_names:
#     file_path = os.path.join('corpus', file_name)

#     with open(file_path, 'r') as infile:
#         file_content = file_content + infile.read().replace('\n', '')

# %%
with open('corpus/test.txt', 'r') as infile:
    file_content = infile.read().replace('\n', '')

# %%
from nltk import sent_tokenize
sentences = sent_tokenize(file_content)

# %%
# splitting
training_size = int(len(sentences) * 0.7)
validation_size = int(len(sentences) * 0.1)

training_data = sentences[:training_size]
validation_data = sentences[training_size:training_size + validation_size]
test_data = sentences[training_size + validation_size:]

# %%
import re
from nltk import word_tokenize, ngrams
import string

emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

number_backet_pattern = re.compile(r"\d")
                                
def tokenize_sentences(sentences):
    # convert to lower case
    low_sentences = [sentence.lower() for sentence in sentences]

    # remove number bracket
    cleaned_sentences = [re.sub(number_backet_pattern, "", sentence) for sentence in low_sentences]


    # remove emoji
    no_emoji_sentences = [re.sub(emoji_pattern, "", sentence) for sentence in cleaned_sentences]

    # split each sentence into tokens
    tokens_2d = [word_tokenize(sentence) for sentence in no_emoji_sentences]

    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    no_punc_tokens = []
    for tokens_1d in tokens_2d:
        no_punc_tokens.append([token.translate(translator) for token in tokens_1d])

    # # remove empty string
    no_empty_tokens = []
    for token_1d in no_punc_tokens:
        no_empty_tokens.append([token for token in token_1d if token != ''])

    # generate n-grams
    uni_grams = []
    bi_grams = []
    tri_grams = []
    four_grams = []

    for token_1d in no_empty_tokens:
        uni_grams.append(list(ngrams(token_1d, n=1)))
        bi_grams.append(list(ngrams(token_1d, n=2)))
        tri_grams.append(list(ngrams(token_1d, n=3)))
        four_grams.append(list(ngrams(token_1d, n=4)))

    # return a dictionary containing all lists of n-grams
    return {
        'sentences': no_empty_tokens,
        'uni_grams': uni_grams,
        'bi_grams': bi_grams,
        'tri_grams': tri_grams,
        'four_grams': four_grams,
    }

# %%
# handle unknown word

# %%
tokenized_sent = tokenize_sentences(training_data)

# %%
# if (__name__ == '__main__'):
#     for l in tokenized_sent['four_grams']:
#         print(l)

# %%
# from nltk.probability import FreqDist


# freq_uni = FreqDist()
# freq_bi = FreqDist()
# freq_tri = FreqDist()
# freq_four = FreqDist()

# # count uni-grams
# for l in tokenized_sent['uni_grams']:
#     for uni_gram in l:
#         freq_uni[uni_gram] = freq_uni[uni_gram] + 1

# # count bi-grams
# for l in tokenized_sent['bi_grams']:
#     for bi_gram in l:
#         freq_bi[bi_gram] = freq_bi[bi_gram] + 1
        
# # count bi-grams
# for l in tokenized_sent['tri_grams']:
#     for tri_gram in l:
#         freq_tri[tri_gram] = freq_tri[tri_gram] + 1

# # count bi-grams
# for l in tokenized_sent['four_grams']:
#     for four_gram in l:
#         freq_four[four_gram] = freq_four[four_gram] + 1


# %%
# if (__name__ == '__main__'):
#     for word in freq_four:
#         print(f'{word}: {freq_four[word]}')

# %%
def tokenize_words(data):
    words = []
    sentences = tokenize_sentences(data)['sentences']
    for sentence in sentences:
        words.append('<s>')
        words.extend(sentence)
        words.append('</s>')

    # create n-grams
    uni_grams = []
    bi_grams = []
    tri_grams = []
    four_grams = []

    uni_grams.extend(list(ngrams(words, n=1)))
    bi_grams.extend(list(ngrams(words, n=2)))
    tri_grams.extend(list(ngrams(words, n=3)))
    four_grams.extend(list(ngrams(words, n=4)))

    return {
        'sentences': words,
        'uni_grams': uni_grams,
        'bi_grams': bi_grams,
        'tri_grams': tri_grams,
        'four_grams': four_grams,
    } 
    

# %%
# if (__name__ == '__main__'):
#     for l in tokenize_words(training_data)['four_grams']:
#         print(l)

# %%
# tokenize_words()['uni_grams']
    

# %%
from nltk.probability import FreqDist


freq_uni = FreqDist()
freq_bi = FreqDist()
freq_tri = FreqDist()
freq_four = FreqDist()

tokenized_words = tokenize_words(training_data);

# count uni-grams
for uni_gram in tokenized_words['uni_grams']:
    freq_uni[uni_gram] = freq_uni[uni_gram] + 1

# count bi-grams
for bi_gram in tokenized_words['bi_grams']:
    freq_bi[bi_gram] = freq_bi[bi_gram] + 1
        
# count bi-grams
for tri_gram in tokenized_words['tri_grams']:
    freq_tri[tri_gram] = freq_tri[tri_gram] + 1

# count bi-grams
for four_gram in tokenized_words['four_grams']:
    freq_four[four_gram] = freq_four[four_gram] + 1


# %%
# if (__name__ == '__main__'):
    # print(tokenize_sentences(sentences)['sentences'])
    # print(freq_uni.get(('is',), 0))
    # for word in freq_four:
    #     print(f'{word}: {freq_four[word]}')


