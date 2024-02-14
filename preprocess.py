# %%
from nltk import sent_tokenize, word_tokenize, ngrams
import re
import string
from nltk.probability import FreqDist
from nltk.corpus import stopwords


# %%
class TextPreprocessor:
    def __init__(self) -> None:
        with open('corpus/corpus.txt', 'r') as infile:
            file_content = infile.read().replace('\n', '')
            
        # split the file content into sentences
        self.sentences = sent_tokenize(file_content)
        
        # ---------- Train, Validation, Test ----------
        # size of training data
        self.training_size = int(len(self.sentences) * 0.7)
        self.training_data = self.sentences[:self.training_size]

        # size of validation data
        self.validation_size = int(len(self.sentences) * 0.1)
        self.validation_data = self.sentences[self.training_size:self.training_size + self.validation_size]

        self.test_data = self.sentences[self.training_size + self.validation_size:]
        
        # ---------- Call necessary methods ----------
        # self.find_unknown_word(self.tokenize_words(self.test_data))
        self.create_freq_n_gram(self.training_data)
        pass

    def tokenize_words(self, sentences):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        number_bracket_pattern = re.compile(r"\d")
        
        def get_lower(sentences: list[str]): return [sentence.lower() for sentence in sentences]
        
        # convert to lower case
        preprocessed_tokens = [sentence.lower() for sentence in sentences]

        # remove number bracket
        preprocessed_tokens = [re.sub(number_bracket_pattern, "", sentence) for sentence in preprocessed_tokens]

        # remove emoji
        preprocessed_tokens = [re.sub(emoji_pattern, "", sentence) for sentence in preprocessed_tokens]

        # split each sentence into tokens: [['token', ''token'], ['token', 'token'], ...]
        proprocessed_tokens = [word_tokenize(sentence) for sentence in preprocessed_tokens]
        
        # remove stopwords
        en_stopwords = stopwords.words('english')
        no_stopword_tokens = []
        for tokens_1d in proprocessed_tokens:
            no_stopword_tokens.append([token for token in tokens_1d if token not in en_stopwords])
            
        # remove articles
        articles = ['the', 'a', 'an']
        no_article_tokens = []
        for tokens_1d in no_stopword_tokens:
            no_article_tokens.append([token for token in tokens_1d if token not in articles])

        # remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        no_punc_tokens = []
        for tokens_1d in no_article_tokens:
            no_punc_tokens.append([token.translate(translator) for token in tokens_1d])

        # # remove empty string
        no_empty_tokens = []
        for token_1d in no_punc_tokens:
            no_empty_tokens.append([token for token in token_1d if token != ''])
            
        # convert into token list: ['token', 'token']
        tokens = []
        for sentence in no_empty_tokens:
            # add start and end tokens to identify sentence boundary
            tokens.append('<s>')
            
            # add words in each sentence into tokens
            tokens.extend(sentence)
            tokens.append('</s>')

        # create n-grams
        uni_grams = []
        bi_grams = []
        tri_grams = []
        four_grams = []

        uni_grams.extend(list(ngrams(tokens, n=1)))
        bi_grams.extend(list(ngrams(tokens, n=2)))
        tri_grams.extend(list(ngrams(tokens, n=3)))
        four_grams.extend(list(ngrams(tokens, n=4)))

        return {
            'sentences': tokens,
            'uni_grams': uni_grams,
            'bi_grams': bi_grams,
            'tri_grams': tri_grams,
            'four_grams': four_grams,
        }
        
    def create_freq_n_gram(self, training_data):
        # create a frequency distribution for each n-gram
        self.freq_uni = FreqDist()
        self.freq_bi = FreqDist()
        self.freq_tri = FreqDist()
        self.freq_four = FreqDist()

        tokenized_words = self.tokenize_words(training_data);
        
        # count uni-grams
        for uni_gram in tokenized_words['uni_grams']:
            self.freq_uni[uni_gram] = self.freq_uni[uni_gram] + 1

        # count bi-grams
        for bi_gram in tokenized_words['bi_grams']:
            self.freq_bi[bi_gram] = self.freq_bi[bi_gram] + 1
                
        # count bi-grams
        for tri_gram in tokenized_words['tri_grams']:
            self.freq_tri[tri_gram] = self.freq_tri[tri_gram] + 1

        # count bi-grams
        for four_gram in tokenized_words['four_grams']:
            self.freq_four[four_gram] = self.freq_four[four_gram] + 1
            
    # handle unknown word
    # def find_unknown_word(self, training:list, testing:list):
    #     training = self.tokenize_words(training)
    #     for word in training:
    #         if word not in testing:
    #             training[training.index(word)] = '<UNK>'
                
    #     for word in testing:
    #         if word not in training:
    #             testing[testing.index(word)] = '<UNK>'
    #         i = i + 1


