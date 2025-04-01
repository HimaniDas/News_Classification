from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
from pre_processor import pre_process

def read_files():

    #we declared two list to receive the values
    train_data = []
    test_data = []


    with open('News_Classification/train.txt', 'r') as train_file:
        for line in train_file:
            train_data.append(line)

    with open('News_Classification/dev.txt', 'r') as test_file:
        for line in test_file:
            test_data.append(line)

    #from here the dataset and test data will be returned to main function
    return train_data, test_data

"""this is our label sepreating function
in this function the labels and documents"""
def separate_labels(data):
    documents = []
    labels = []

    for line in data:

        """this will split the whole document where it gets a tab"""
        splitted_line = line.split('\t', 1)
        # separate the labels and examples (docs) in different list
        labels.append(splitted_line[0])
        documents.append(splitted_line[1])

    return documents, labels

# we use a dummy function as tokenizer and preprocessor,
def identity(X):
    return X

#this is our vectorizer function
def vectorization(tfidf):
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity,
                              analyzer='char',
                              ngram_range=(4,6))
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity,
                              analyzer='char',
                              ngram_range=(4,7))
    return vec


"""this is our function for naive bayes classifier"""
def Naive_Bayes(train_docs, train_lbls, test_docs, test_lbls):

    # since the texts are already preprocessed and tokenized.

    """here the vectorizer will be called and the selected vecotorizer
    will be passed as parameter."""
    tfidf = False
    vec = vectorization(tfidf)

    # combine the vectorizer with a Naive Bayes classifier and send it through pipeline
    classifier = Pipeline([('vec', vec),
                           ('cls', MultinomialNB())])

    """though we took a variable classifier where vectorizer as vec and multinomialNB as cls were assigned,
        still classifier is empty as there is no data to be trained. so we used fit method to fit all the data in classifier"""
    classifier.fit(train_docs, train_lbls)

    #predict is for predicting label for document test data by using predict method
    predict = classifier.predict(test_docs)

    """generate accuracy report"""
    # Compare the accuracy of the output (predict) with the class labels of the original test set (test_lbls)
    print("Accuracy = ", accuracy_score(test_lbls, predict))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(test_lbls , predict, labels=classifier.classes_, target_names=None, sample_weight=None,
                                digits=3))
    # confusion matrix is another better way to evaluate the performance of the classifier we took
    print('Confusion Matrix : ')
    print(confusion_matrix(test_lbls, predict))

def SVM(train_docs, train_lbls, test_docs, test_lbls):
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.

    tfidf = True
    vec = vectorization(tfidf)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                           ('cls', svm.SVC(kernel="linear",
                                           C=1.0))])


    # COMMENT THIS
    classifier.fit(train_docs, train_lbls)


    # COMMENT THIS
    predict = classifier.predict(test_docs)
    # print(predict)


    """generate accuracy report"""
    # Compare the accuracy of the output (predict) with the class labels of the original test set (test_lbls)
    print("Accuracy = ", accuracy_score(test_lbls, predict))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(test_lbls , predict, labels=classifier.classes_, target_names=None, sample_weight=None,
                                digits=3))
    # confusion matrix is another better way to evaluate the performance of the classifier we took
    print('Confusion Matrix : ')
    print(confusion_matrix(test_lbls, predict))


    # file = open("labels.txt", mode="w")
    # file.close()
    # for label in predict:
    #     file = open("labels.txt", "a", encoding='utf-8')
    #     file.write(str(label))
    #     file.write("\n")
    # file.close()



"""This is our user defined function.
From here all the functions will be called.
"""

def main():
    print('Reading The Dataset....')

    #here the read_file function will be called.
    train_data, test_data = read_files()

    #here the labels and documents of both trainset and test set will be seperated.
    train_docs, train_lbls = separate_labels(train_data[:2000])
    test_docs, test_lbls = separate_labels(test_data[:2000])

    print('\nPreprocessing....')
    # only tokenizing the documents

    """form here the pre_process from pre_processor python file will be called
    to do pre_process the whole data"""
    preprocessed_train_docs = pre_process(train_docs)
    preprocessed_test_docs = pre_process(test_docs)


    """from here the naive byse function will be called 
    to run our naive bayes classifier"""
    print('\nTraining The Naive Bayes Classifier....\n\n')
    Naive_Bayes(train_docs=preprocessed_train_docs, train_lbls=train_lbls,
                test_docs=preprocessed_test_docs, test_lbls=test_lbls)


    """from here the naive byse function will be called 
    to run our SVM classifier"""
    print('\nTraining The SVM Classifier....\n\n')
    SVM(train_docs=preprocessed_train_docs, train_lbls=train_lbls,
                test_docs=preprocessed_test_docs, test_lbls=test_lbls)




"""This is our main function.
From here the user defined main will be called."""
if __name__ == "__main__":
    main()
