from flask import Flask,render_template,request
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
model_name = 'bert-base-nli-mean-tokens' # using the MEAN pooling strategy for CLS tokens
model = SentenceTransformer(model_name)


def similarity(s1, s2):
    """
    This function takes vector forms of sentences and calculates cosine similarity [0-1]

    """

    product = np.dot(s1, s2)  # dot product between vectors of two sentences
    norm_s1 = norm(s1)  # calculate norm of A
    norm_s2 = norm(s2)  # calculate norm of B
    return product / (norm_s1 * norm_s2)  # cosine similarity = Dot product/(Norm(A) * Norm(B))


def sentence_similarity(text_1, text_2):
    """
    This function converts the input text to vectors of 768 dimensions and finds similarity between them.

    """

    vector_1 = model.encode(text_1)
    vector_2 = model.encode(text_2)
    sim_score = similarity(vector_1, vector_2)

    return ("Similarity score :" +  str(sim_score))

app = Flask(__name__)
@app.route('/')
def form_sentence():
    return render_template('form_sentence.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['sent1']
    text2 = request.form['sent2']
    a = sentence_similarity(text, text2)
    return a


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

