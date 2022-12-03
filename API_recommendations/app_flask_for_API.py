
# Start importing relevant librairies
# Import libraries
import os
import pandas as pd
import numpy as np
from time import time
from random import randint
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from implicit.lmf import LogisticMatrixFactorization
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k, AUC_at_k
import pickle
import flask
from flask import jsonify


clicks = pd.read_csv("https://github.com/archiducarmel/p9-oc/releases/download/clicks/clicks.csv")

MODEL_PATH = "./recommender.model"
if not os.path.exists(MODEL_PATH):
    os.system("wget https://github.com/archiducarmel/p9-oc/releases/download/clicks/recommender.model")


app = flask.Flask(__name__)

# This is the route to the API
@app.route("/")
def home():
    return "Welcome on the recommendation API ! "

@app.route("/get_recommendation/<id>", methods=["POST", "GET"])
def get_recommendation(id):

    recommendations = get_cf_reco(clicks, int(id), csr_item_user, csr_user_item, model_path=MODEL_PATH, n_reco=5, train=False)
    data = {
            "user" : id,
            "recommendations" : recommendations,
        }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=8080)
