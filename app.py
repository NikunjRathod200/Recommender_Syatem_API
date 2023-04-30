from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
        return send_file('movie.html',mimetype='text/html')

@app.route('/movie', methods=['GET'])#'''This function is for content based recommender system results'''
def recommend_movies():
        res = recommendation.results(request.args.get('title'))
        return jsonify(res)


@app.route('/user', methods=['GET'])#'''This function is for collaborative recommender system results.'''
def recommend_movie():
        id = request.args.get('id')#user id should be integer value
        try:
            id = int(id)
        except:
            return jsonify({'error': 'id must be an integer'})
        res = recommendation.coll(id)
        return jsonify(res)


if __name__ == '__main__':
        app.run(port=5000, debug=True)
