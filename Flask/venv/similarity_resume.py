# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_community.document_loaders import PyPDFLoader


# def similarity_checker_tfidf(job_description, resumes_folder):

#     # Path to the folder containing multiple resume PDFs
#     resumes_folder = "C:\\Data\\Python\\Ammar\\Flask\\venv\\Resumes"
#     similarity_scores = []  # To store the scores for all resumes

#     # Iterate through all PDF files in the folder
#     for resume_file in os.listdir(resumes_folder):
#         if resume_file.endswith(".pdf"):  # Only process PDF files
#             resume_path = os.path.join(resumes_folder, resume_file)

#             # Load the resume PDF and extract its text
#             loader = PyPDFLoader(resume_path)
#             resume_text = ""
#             for page in loader.load():
#                 resume_text += page.page_content  # Concatenate the text of all pages

#             # Combine job description and resume into a corpus for TF-IDF vectorization
#             corpus = [job_description, resume_text]

#             # Initialize the TF-IDF Vectorizer
#             vectorizer = TfidfVectorizer()

#             # Transform the corpus into TF-IDF vectors
#             tfidf_matrix = vectorizer.fit_transform(corpus)

#             # Calculate the cosine similarity between the job description (index 0) and resume (index 1)
#             similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

#             # Append the similarity score and resume file name to the result
#             similarity_scores.append((resume_file, similarity[0][0]))

#     # Print or return the list of scores for all resumes
#     for resume_file, score in similarity_scores:
#         print(f"Resume: {resume_file}, Similarity Score (TF-IDF): {score}")

#     return similarity_scores

# # Job description text
# job_description = """
# We want to hire a Machine Learning Engineer for our Company.
# Responsibilities include developing and deploying models, working with large datasets, and collaborating with the team.
# Skills needed: Python, Machine Learning, Deep Learning, Data Science, and cloud computing.
# """



# # Run similarity checker with TF-IDF for all resumes in the folder
# similarity_checker_tfidf(job_description)



from flask import Flask, request, jsonify
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)

def similarity_checker_tfidf(job_description, resumes_folder):
    similarity_scores = []  # To store the scores for all resumes

    # Iterate through all PDF files in the folder
    for resume_file in os.listdir(resumes_folder):
        if resume_file.endswith(".pdf"):  # Only process PDF files
            resume_path = os.path.join(resumes_folder, resume_file)

            # Load the resume PDF and extract its text
            loader = PyPDFLoader(resume_path)
            resume_text = ""
            for page in loader.load():
                resume_text += page.page_content  # Concatenate the text of all pages

            # Combine job description and resume into a corpus for TF-IDF vectorization
            corpus = [job_description, resume_text]

            # Initialize the TF-IDF Vectorizer
            vectorizer = TfidfVectorizer()

            # Transform the corpus into TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Calculate the cosine similarity between the job description (index 0) and resume (index 1)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            # Append the similarity score and resume file name to the result
            similarity_scores.append((resume_file, similarity[0][0]))

    return similarity_scores

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    data = request.json
    job_description = data.get('job_description', '')
    
    # Path to the folder containing multiple resume PDFs
    resumes_folder = "C:\\Data\\Python\\Ammar\\Flask\\venv\\Resumes"
    
    # Run similarity checker with TF-IDF for all resumes in the folder
    similarity_scores = similarity_checker_tfidf(job_description, resumes_folder)

    # Prepare response data
    response_data = [{"resume": resume_file, "similarity_score": score} for resume_file, score in similarity_scores]
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
