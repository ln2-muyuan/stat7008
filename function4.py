import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
import chardet

def function4():
    folder_path = 'extracted_text'

    # test if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    # get all text files in the folder
    txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    # read text data from files
    all_text = []
    for txt_file in txt_files:
        try:
            with open(txt_file, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(txt_file, 'r', encoding=encoding) as f:
                text = f.read().strip()
                if text:
                    all_text.append(text)
                else:
                    print(f"Warning: File {txt_file} is empty.")
        except Exception as e:
            print(f"Error reading file {txt_file}: {e}")

    if not all_text:
        raise ValueError("No valid text data found. Please check your files.")

    # generate TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_text)

    # generate labels for Naive Bayes model
    y = [i % 2 for i in range(len(all_text))] 

    # train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)

    # use agglomerative clustering to group the text files
    clustering = AgglomerativeClustering(n_clusters=3)  
    cluster_labels = clustering.fit_predict(X.toarray())  

    # use SVM to classify the text files
    svm_model = SVC(probability=True)
    svm_model.fit(X, y)

    # predict the class probabilities for each file
    y_pred_proba = model.predict_proba(X)
    for i, file in enumerate(txt_files):
        print(f'File: {file}, Class probabilities: {y_pred_proba[i]}')

    # output the cluster labels for each file
    for i, label in enumerate(cluster_labels):
        print(f'File: {txt_files[i]} belongs to cluster {label}')

