import numpy as np 
import pandas as pd
import re
import time
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class K_Nearest_Neighbor:
    def __init__(self,k):
        self.K = k
    
    #read data from the .dat from  both train and test files
    def read_data(self):
        Train_data_file = pd.read_csv("train_data.dat", sep='\t', names=["Class","Reviews"], usecols=range(2))
        Test_data_file = pd.read_fwf("test_data.dat", sep='\t' ,names=["Reviews"], usecols=range(1),skip_blank_lines=False, na_filter=False)
        
        #Drop the blank lines from training file
        Train_data_file = Train_data_file.dropna()
        return (Train_data_file,Test_data_file)
    
    #remove punctuations
    def remove_punctuation(self, text_data):    
        no_punct_text = re.sub("[^a-zA-Z]"," ",str(text_data))
        return no_punct_text
    
    #remove stopwords
    def remove_stopwords(self, text_data):
        stop_words = set(stopwords.words('english'))        
        filtered_words = [word for word in text_data if word not in stop_words]
        return filtered_words
    
    #word lemmatization
    def lemmatize_words(self, text_data):    
        lemmatizer = WordNetLemmatizer()        
        filtered_lemmatized_words = [lemmatizer.lemmatize(word) for word in text_data]            
        return filtered_lemmatized_words
    
    #words stemming
    def stemming_words(self, text_data):
        stemming = PorterStemmer()
        filtered_stemmed_words = ",".join([stemming.stem(word) for word in text_data])
        return filtered_stemmed_words
    
    #Snippet is for text cleaning
    def text_preprocess(self, text_data):        
    
        #remove punctuation
        text_data["Reviews"] = text_data["Reviews"].apply(lambda text : self.remove_punctuation(text))                  
        
        #tokenize words    
        tokenizer = RegexpTokenizer(r'\w+')
        text_data["Reviews"] = text_data["Reviews"].apply(lambda text : tokenizer.tokenize(text.lower()))    
        
        #removal of stopwords
        text_data["Reviews"] = text_data["Reviews"].apply(lambda text : self.remove_stopwords(text))    
        
        #lemmatization of words
        text_data["Reviews"] = text_data["Reviews"].apply(lambda text : self.lemmatize_words(text))    
        
        #stemming of words
        text_data["Reviews"] = text_data["Reviews"].apply(lambda text : self.stemming_words(text))    
            
        return text_data
    
    #Vectorization of text using TF-IDF vectorizer
    def TFIDF_vectorizer_cosine_similarity(self, train, test):        
        vectorizer = TfidfVectorizer()            
        train_vectors = vectorizer.fit_transform(train)        
        test_vectors = vectorizer.transform(test)
        text_cosine_similarity = cosine_similarity(test_vectors, train_vectors)            
        return text_cosine_similarity
        
   #KNN implementation
    def K_Nearest_Neighbor_Classifier(self, text_cosine_similarity, test_file, train_label, submission):
        pred_labels = list()
        k = self.K
        for sentence_vector in text_cosine_similarity:                    
            #get indexes of sorted vector
            n_indexes = sentence_vector.argsort()
            n_indexes = n_indexes[:-k:-1]
            #get the labels of all the training data in a new list
            train_labels_list = list()
            for n_index in n_indexes:
                train_labels_list.append(train_label[n_index])
    
            #here we are taking weighted approach so if label is -1 then assign -ve sign to our cosine similarity
            #then take sum of everything and if the value is +ve assign +1 and if the value is +ve then assign -1
    
            signed_similarity_scores = list()            
            for similarity_score, label in zip(sentence_vector, train_labels_list):
                if(label == -1):
                    signed_similarity_scores.append(-similarity_score)
                else:                
                    signed_similarity_scores.append(similarity_score)
    
            total_score_sum = sum(signed_similarity_scores)            
    
            if(total_score_sum > 0):
                if(submission == False):
                    pred_labels.append(1)
                else:
                    pred_labels.append("+1")
            else:            
                pred_labels.append(-1)
        return pred_labels        
    #KNN implementation END

#Start of execution===========================================================>
#This is just to check how much time it is taking for execution
execution_start_time = time.time()
k = 35
knn = K_Nearest_Neighbor(k)

#Step - 1 Read data
train_data_file, test_data_file = knn.read_data()

#Step-2 data preprocessing
#this is the data from train.dat file needs to clean
cleaned_train_data = knn.text_preprocess(train_data_file)

#this is the data from test.dat file needs to clean
cleaned_test_data = knn.text_preprocess(test_data_file) 

#Step-3 Split train-test data 80% training and 20% test data from cleaned_train_data
X_train_data, X_test_data, y_train_label, y_test_label = train_test_split(cleaned_train_data, cleaned_train_data["Class"], train_size=0.80, shuffle = True)

#Step -4 Vectorize the train and test data using TF-IDF vectorizer and calculate cosine_similarities
cosine_similarity_scores = knn.TFIDF_vectorizer_cosine_similarity(X_train_data["Reviews"],X_test_data["Reviews"])
        
#Step-5 call KNN function on training data
y_prediction = knn.K_Nearest_Neighbor_Classifier(cosine_similarity_scores, X_test_data["Reviews"].to_numpy(), y_train_label.to_numpy(),False)

#works best for k=35 for test data after split
print("Accuracy using train_test split :\n",accuracy_score(list(y_test_label), y_prediction))

#Vectorize complete train file and test file
textual_cos_similarity = knn.TFIDF_vectorizer_cosine_similarity(cleaned_train_data["Reviews"], cleaned_test_data["Reviews"])

#call KNN on all data from test file
final_predictions = knn.K_Nearest_Neighbor_Classifier(textual_cos_similarity , cleaned_test_data["Reviews"].to_numpy(), cleaned_train_data["Class"].to_numpy(),True)

#We will just check the final predicted classes    
print("Final predictions of review from test file: \n", final_predictions)

#Now to save the predicted classes into a file
final_predictions_df = pd.DataFrame(final_predictions) 
final_predictions_df.to_csv("Final_predictions.csv",columns=None,index=None,header=None)

execution_end_time = time.time()
#Total execution time:
print("Total code execution time:",execution_end_time - execution_start_time," seconds" )

#End of execution===========================================================>
