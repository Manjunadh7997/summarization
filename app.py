import os
import docx2txt
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

app = Flask(__name__)

def summarize_word_document(input_doc_path, output_doc_path, max_summary_length=500, keyword_density=0.7,
                            top_n_sentences=5, sentence_length_filter=(10, 20)):
    # Read the Word document
    text = docx2txt.process(input_doc_path)

    # Tokenize the content into sentences
    sentences = sent_tokenize(text)

    # Filter sentences based on length
    filtered_sentences = [sentence for sentence in sentences if
                          sentence_length_filter[0] <= len(sentence.split()) <= sentence_length_filter[1]]

    # Calculate TF-IDF vectors for the sentences
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(filtered_sentences)

    # Calculate keyword density for each sentence
    keyword_density_scores = [(sum(word in sentence.lower() for word in text.lower().split()) / len(sentence.split()), sentence)
                              for sentence in filtered_sentences]

    # Select top N sentences based on keyword density
    selected_sentences = [sentence for density, sentence in sorted(keyword_density_scores, reverse=True)[:top_n_sentences]
                          if density >= keyword_density]

    # Calculate pairwise cosine similarity between sentences
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Select top N sentences based on cosine similarity
    summary_sentences = []
    total_length = 0
    for i in range(len(filtered_sentences)):
        index = cosine_similarities.sum(axis=1).argmax()
        if total_length + len(filtered_sentences[index]) <= max_summary_length:
            summary_sentences.append(filtered_sentences[index])
            total_length += len(filtered_sentences[index])
        else:
            break
        cosine_similarities[index] = -1  # Remove selected sentence from consideration in the next iteration

    # Join selected sentences to form the summary
    summary = ' '.join(summary_sentences)

    # Save summary in a separate Word document
    doc = Document()
    doc.add_paragraph(summary)
    doc.save(output_doc_path)

    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_doc_path = data['inputPath']
    output_doc_path = data['outputPath']
    max_summary_length = int(data['maxSummaryLength'])
    keyword_density = float(data['keywordDensity'])
    top_n_sentences = int(data['topNSentences'])
    min_sentence_length = int(data['minSentenceLength'])
    max_sentence_length = int(data['maxSentenceLength'])
    
    summary = summarize_word_document(input_doc_path, output_doc_path, max_summary_length, keyword_density,
                                      top_n_sentences, (min_sentence_length, max_sentence_length))

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
