import nltk
import string
import random
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppressing warnings
warnings.filterwarnings("ignore")

# Downloading necessary NLTK packages
nltk.download('punkt')  # for tokenization
nltk.download('wordnet')  # for lemmatization
nltk.download('omw-1.4')  # for the Open Multilingual Wordnet

f = open('output.txt', 'r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
# Tokenizing the document into sentences and words
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

# Initializing the WordNet lemmatizer
# from nltk.stem import WordNetLemmatizer

# the provided code sets up a WordNetLemmatizer instance and defines a function that takes a list of tokens as input and returns a list of lemmatized tokens.

lemmer = nltk.stem.WordNetLemmatizer()

# Function to lemmatize tokens
def LemToken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Creating a dictionary of punctuations to be removed
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

# translate(): remove punctuation from text

def LemNormalize(text):
    return LemToken(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Pre-defined greetings and responses
greet_inputs = ('hello', 'hi', 'wassup', 'hey')
greet_responses = ('hi', 'hey!', 'hey there!', 'hola user')

# Function to check if the input sentence is a greeting and generate a random response
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)


# Function to generate a response to the user input using TF-IDF and cosine similarity
# TF-IDF (Term Frequency-Inverse Document Frequency) approach and cosine similarity to select an
# appropriate response based on the similarity of the user input to previous system responses.

def response(user_response):
    robo1_response = ''
    sentence_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=r'(?u)\b\w\w+\b')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]  # Index of the most similar element
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I am sorry, I am unable to understand you"
        return robo1_response
    else:
        robo1_response = robo1_response + sentence_tokens[idx]
        return robo1_response


# Main execution part
flag = True
print('Bot: Hello, I am TAM chatbot, How can I help you with?')
while flag:
    user_response = input('You: ')
  # lower user responses
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            flag = False
            print('Bot: You are welcome')
        else:
            if greet(user_response) is not None:
                print('Bot:', greet(user_response))
            else:
              # if the user input is not a greeting, then generate a response using TF-IDF and add
              # it to the array of vector for comparing in TFIDF

                word_tokens += nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                bot_response = response(user_response)
                print('Bot:', bot_response[0] if isinstance(bot_response, list) else bot_response)
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Goodbye!!')

