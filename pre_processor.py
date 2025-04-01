import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#make texts lowercase
def text_lowercase(text):
    return text.lower()


# Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result



# remove punctuation
def remove_punctuation(text):
    # map punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation), '')
    return text.translate(translator)

    # for punc in string.punctuation:
    #     translator = str.maketrans(punc, ' ', '')
    #     text = text.translate(translator)

    # return text


# remove whitespace from text
def remove_whitespace(text):
    return " ".join(text.split())


def tokenie_sentence(text):
    word_tokens = word_tokenize(text)
    return word_tokens


# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = tokenie_sentence(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text



def pre_process(documents):
    # # combined_doc = []
    lowercased_doc = []
    for sent in documents:
        lowercased_sen = text_lowercase(sent)
        lowercased_doc.append(lowercased_sen)

    # print("Output of Lowercasing Operation:\n", temp_doc)
    # return  lowercased_doc

    documents = lowercased_doc
    remove_number_doc = []
    for sent in documents:
        remove_number_doc.append(remove_numbers(sent))
    # return  remove_number_doc

    # print("Output of Removed Number Operation:\n", temp_doc)
    documents = remove_number_doc
    remove_punc_doc = []
    for sent in documents:
        remove_punc_doc.append(remove_punctuation(sent))

    # print("Output of Remove Punctuations Operation:\n", temp_doc)

    # return remove_punc_doc

    documents = remove_punc_doc
    temp_doc = []
    for sent in documents:
        temp_doc.append(remove_whitespace(sent))

    # print("Output of Removing Multiple Whitespaces Operation:\n", temp_doc)
    return temp_doc


