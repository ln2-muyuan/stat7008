import requests
import base64
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to fetch file content from GitHub
def fetch_github_file_content(username, repository, filepath):
    url = f'https://api.github.com/repos/{username}/{repository}/contents/{filepath}'
    response = requests.get(url)
    
    if response.status_code == 200:
        content = response.json().get('content')
        if content:
            return base64.b64decode(content).decode('utf-8')
    
    return None

# Function to extract keywords using TF-IDF
def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [(feature_names[i], tfidf_scores[i]) for i in sorted_indices[:top_n]]
    return top_keywords

# Function to generate word cloud from keywords
def generate_wordcloud(keywords, company_name):
    word_freq = {keyword: score for keyword, score in keywords}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {company_name}')
    wordcloud_filename = f'wordcloud_{company_name.replace(" ", "_")}.png'
    plt.savefig(wordcloud_filename)
    plt.show()

# Function to generate bar chart for top frequency words
def generate_bar_chart(top_words, company_name):
    words, frequencies = zip(*top_words)
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(words)), frequencies, align='center', color='skyblue')
    plt.yticks(range(len(words)), words)
    plt.gca().invert_yaxis()  # Invert y-axis to display the most frequent word at the top
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.title(f'Top 30 Most Frequent Words for {company_name}')
    bar_chart_filename = f'barchart_{company_name.replace(" ", "_")}.png'
    plt.savefig(bar_chart_filename)
    plt.show()

# Function to save TF-IDF values to a text file
def save_tfidf_values(tfidf_values, company_name):
    tfidf_filename = f'tfidf_values_{company_name.replace(" ", "_")}.txt'
    with open(tfidf_filename, 'w') as file:
        for keyword, score in tfidf_values:
            file.write(f'{keyword}: {score}\n')

# Function to analyze text file and save outputs
def analyze_text_file(filename):
    path = "" + filename
    company_name = filename.split('.')[0]  # Extract company name from filename
    print("Analyzing file: " + filename)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    keywords = extract_keywords_tfidf(content, top_n=20)
    generate_wordcloud(keywords, company_name)
    generate_bar_chart(keywords[:30], company_name)
    save_tfidf_values(keywords, company_name)
    
    return keywords

# Get the current working directory
current_directory = os.getcwd()

# Get a list of all files in the current working directory
file_names = [f for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]

# Perform analysis on each file and save outputs
all_keywords = []
for filename in file_names:
    keywords = analyze_text_file(filename)
    all_keywords.extend(keywords)

# Combine all keywords into a single set
combined_keywords = list(set(all_keywords))

# Generate word cloud and bar chart for the combined keywords
generate_wordcloud(combined_keywords, "Combined Analysis")
generate_bar_chart(combined_keywords[:30], "Combined Analysis")
save_tfidf_values(combined_keywords, "Combined Analysis")
