# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import string

# # Download NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(text)
#     filtered_text = [word for word in word_tokens if word not in stop_words]
#     return ' '.join(filtered_text)

# # Load the CSV file
# df = pd.read_csv("D:\Internship_portaL\Flipkart_data.csv")

# # Clean the descriptions
# df['cleaned_description'] = df['description'].apply(clean_text)

# print(df.head())

import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

# Ensure the NLTK resources are downloaded
nltk.download('stopwords')

# Define the text cleaning function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load your DataFrame
df = pd.read_csv("D:\Internship_portaL\LinkedIn_jobs_data.csv")

# Print the columns to verify 'description' exists
print("Columns before cleaning:", df.columns)

# Clean the column names
df.columns = df.columns.str.strip().str.lower()

# Print the columns again to verify
print("Columns after cleaning:", df.columns)

# Check if 'description' column exists
if 'description' in df.columns:
    # Apply the clean_text function to the 'description' column
    df['cleaned_description'] = df['description'].apply(clean_text)
    print("Cleaned descriptions added.")
else:
    print("Column 'description' not found in the DataFrame.")

print(df.head())

df.to_csv('preprocessed_csv.csv', index=False)
print("DataFrame saved to 'preprocessed_csv.csv'.")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the processed DataFrame
df = pd.read_csv('preprocessed_csv.csv')

# Check if 'cleaned_description' column exists
if 'cleaned_description' in df.columns:
    # Convert the text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save the cosine similarity matrix to a CSV file
    similarity_df = pd.DataFrame(cosine_sim)
    similarity_df.to_csv('similar.csv', index=False)
    print("Cosine similarity matrix saved to 'similar.csv'.")
    
    # Example: find the most similar descriptions to the first description
    similarity_scores = list(enumerate(cosine_sim[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Print the top 5 most similar descriptions
    print("Top 5 most similar descriptions to the first one:")
    for i in similarity_scores[1:6]:
        print(f"Index: {i[0]}, Similarity Score: {i[1]}")
        print(df['cleaned_description'].iloc[i[0]])
        print()
else:
    print("Column 'cleaned_description' not found in the DataFrame.")
