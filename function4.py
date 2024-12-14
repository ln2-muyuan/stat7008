import os

def function4():
    folder_path = 'extracted_text'
    txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    all_text = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            text = f.read()
            all_text.append(text)
            from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_text)
    y = [0] * len(all_text)  
    model = MultinomialNB()
    model.fit(X, y)

    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters = 3)  # Assume 3 parts
    cluster_labels = clustering.fit_predict(X)

    from sklearn.svm import SVC
    svm_model = SVC()
    svm_model.fit(X, y)

    y_pred_proba = model.predict_proba(X)
    for i, file in enumerate(txt_files):
        print(f'File: {file}, Class probabilities: {y_pred_proba[i]}')

    for i, label in enumerate(cluster_labels):
        print(f'File: {txt_files[i]} belongs to cluster {label}')
