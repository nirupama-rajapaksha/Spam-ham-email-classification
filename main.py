# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:55:03 2018

@author: Nirupama Rajapaksha
"""
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def readHamData():
    ham_emails=list()
    
    with open("SMSSpamCollection.txt","r") as emails:
        content_emails=emails.read()
        emails.close()
    
    for line in re.findall(r'(?<=ham)(.*)(?=\n)',content_emails):
        ham_emails.append(line.strip())
    return ham_emails

def readSpamData():
    spam_emails=list()
    
    with open("SMSSpamCollection.txt","r") as emails:
        content_emails=emails.read()
        emails.close()

    for line in re.findall(r'(?<=spam)(.*)(?=\n)',content_emails):
        spam_emails.append(line.strip())
    return spam_emails
        
def splitingToSentences(fullEmail):
    for item in fullEmail:
        sentences = nltk.sent_tokenize(fullEmail)
    return sentences

def removePunctuation(withPunctuations):
    regex = re.compile('[^a-zA-Z\s\']')
    noPunctuations=regex.sub('', withPunctuations)
    return  noPunctuations

def toLowercase(lowerAndUpper):
    allLowerCase = lowerAndUpper.lower()
    return allLowerCase

def removeStopwords(withStopwords):
    stopWords= stopwords.words('english')
    wordsWithStopwords=regexp_tokenize(withStopwords, "[\w']+")
    sentence=""
    for word in wordsWithStopwords:
        if word not in stopWords:
            if(sentence==""):
                sentence=word
            else:
                sentence=sentence+" "+word
    return sentence


def wordLemmatizing(sentencesToBeLemmitized):
    lemmatizer = WordNetLemmatizer()
    wordsToBeLemmatized=regexp_tokenize(sentencesToBeLemmitized, "[\w']+")
    sentence=""
    for word in wordsToBeLemmatized:
        lemmatizedWord=lemmatizer.lemmatize(word)
        if(sentence==""):
            sentence=lemmatizedWord
        else:
            sentence=sentence+" "+lemmatizedWord
    return sentence

def unigram(sentencesList):
    unigram_list=list()
    for item in sentencesList:
        sentence=regexp_tokenize(item, "[\w']+")
        unigram_list=unigram_list+sentence
    return unigram_list

def bigram(sentencesList):
    bigram_list=list()
    bigrams = [b for l in sentencesList for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    bigram_list=bigrams+bigram_list
    return bigram_list
   
def ham_corpus():

    emails=list()
    email_sentences=list()
    unigrams=list()
    bigrams=list()
    
    print("Making ham corpus")
    print("Reading data and classifying emails to two lists")
    emails=readHamData()
    
    print("Splitting emails to sentences")
    email_list=[splitingToSentences(item) for item in emails]
    for item in email_list:
        email_sentences=email_sentences+item
        
    print("Removing punctuations")
    sentences_without_punctuations=[removePunctuation(item) for item in email_sentences] 
    
    print("Converting to lower case")
    sentences_in_lowerCase=[toLowercase(item) for item in sentences_without_punctuations]
    
    print("Removing stop words")
    sentences_without_stopwords=[removeStopwords(item) for item in sentences_in_lowerCase]
    
    print("Lemmatizing words")
    lemmatized_sentences=[wordLemmatizing(item) for item in sentences_without_stopwords]

    print("Creating unigrams")    
    unigrams=unigram(lemmatized_sentences)
    
    print("Creating bigrams")
    bigrams=bigram(lemmatized_sentences)
    
    return unigrams,bigrams


def spam_corpus():

    emails=list()
    email_sentences=list()
    unigrams=list()
    bigrams=list()
    
    print("Making spam corpus")
    print("Reading data and classifying emails to two lists")
    emails=readSpamData()
    
    print("Splitting emails to sentences")
    email_list=[splitingToSentences(item) for item in emails]
    for item in email_list:
        email_sentences=email_sentences+item
    
    print("Removing punctuations")
    sentences_without_punctuations=[removePunctuation(item) for item in email_sentences]
    
    print("Converting to lower case")
    sentences_in_lowerCase=[toLowercase(item) for item in sentences_without_punctuations]
    
    print("Removing stop words")
    sentences_without_stopwords=[removeStopwords(item) for item in sentences_in_lowerCase]
    
    print("Lemmatizing words")
    lemmatized_sentences=[wordLemmatizing(item) for item in sentences_without_stopwords]

    print("Creating unigrams")    
    unigrams=unigram(lemmatized_sentences)
    
    print("Creating bigrams")
    bigrams=bigram(lemmatized_sentences)
    
    return unigrams,bigrams
 
def uniqueElementsInCorpus(hamOrSpam):
    if(hamOrSpam=="ham"):
        ham_unigram,ham_bigram=ham_corpus()
        uniqueHamUnigrams=set(ham_unigram)
        uniqueHamBigrams=set(ham_bigram)
        return uniqueHamUnigrams,uniqueHamBigrams
    else:
        spam_unigram,spam_bigram=spam_corpus()
        uniqueSpamUnigrams=set(spam_unigram)
        uniqueSpamBigrams=set(spam_bigram)
        return uniqueSpamUnigrams,uniqueSpamBigrams
                
   
def classifier(email):
    
    email_list=splitingToSentences(email)
  
    sentences_without_punctuations=[removePunctuation(item)for item in email_list]    
    sentences_in_lowerCase=[toLowercase(item) for item in sentences_without_punctuations]
    sentences_without_stopwords=[removeStopwords(item) for item in sentences_in_lowerCase]
    lemmatized_sentences=[wordLemmatizing(item) for item in sentences_without_stopwords]
    
    unigrams=unigram(lemmatized_sentences)
    bigrams=bigram(lemmatized_sentences)
    
    ham_unigram,ham_bigram=ham_corpus()
    uniqueHamUnigrams,uniqueHamBigrams=uniqueElementsInCorpus("ham")
    
    ham_probability=1
    
    
    for i in range(len(bigrams)-1):
        ham_probability*=((ham_bigram.count(bigrams[i]))+1)/((ham_unigram.count(unigrams[i]))+len(uniqueHamUnigrams))
        
    spam_unigram,spam_bigram=spam_corpus()
    uniqueSpamUnigrams,uniqueSpamBigrams=uniqueElementsInCorpus("spam")
    
    spam_probability=1
    
    for i in range(len(bigrams)-1):
        spam_probability*=((spam_bigram.count(bigrams[i]))+1)/((spam_unigram.count(unigrams[i]))+len(uniqueSpamUnigrams))
        
    print("Spam probability:",spam_probability)
    print("Ham probability:",ham_probability)
    
    if(ham_probability>spam_probability):
        print("Message is not a spam")
    else:
        print("Message is spam")
       
        
def main():
    email=input("Please enter the email:")
    classifier(email)
    
if __name__ == '__main__':
    main()
    
