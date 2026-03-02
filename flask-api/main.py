import matplotlib
matplotlib.use('Agg')  # use a backend that can make charts without opening a window

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)    # create the flask app
CORS(app)  # allow browser calls from the extension


def preprocess_comment(comment):
    """clean a comment so the model gets better input"""
    try:
        # make all text lowercase so words match better
        comment = comment.lower()

        # remove extra spaces from start and end
        comment = comment.strip()

        # replace line breaks with spaces
        comment = re.sub(r'\n', ' ', comment)

        # remove weird symbols and keep basic punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # remove common filler words but keep key sentiment words
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # convert words to base form like "running" -> "run"
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        app.logger.error(f"Error in preprocessing comment: {e}")
        return comment
    


# load the model from mlflow and vectorizer from local file
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    
    mlflow.set_tracking_uri("http://3.238.38.3:5000/")  
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
   
    return model, vectorizer


# load once at startup so each request is faster
model, vectorizer = load_model_and_vectorizer("my-model", "1", "./bow_vectorizer.pkl")


@app.route('/')
def home():
    # keep a tiny health route to check if api is alive
    return "Sko Buffs!"



@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    # read json data sent from popup
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # clean each comment before turning text into numbers
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # convert text into model features
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # build a dataframe so model gets named feature columns
        feature_names = vectorizer.get_feature_names_out()
        transformed_comments_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
        
        # run sentiment prediction
        predictions = model.predict(transformed_comments_df).tolist()
        
        # convert predictions to string so response type is consistent
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # send comment + sentiment + timestamp back together
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    # do the same prediction flow but without timestamps
    data = request.json
    comments = data.get('comments')
    print("i am the comment: ",comments)
    print("i am the comment type: ",type(comments))
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # clean each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # convert text into numeric features
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # keep feature names so the model input stays aligned
        feature_names = vectorizer.get_feature_names_out()
        transformed_comments_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        # run prediction and convert output to list
        predictions = model.predict(transformed_comments_df).tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # send each comment with its predicted sentiment
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        # read sentiment counts from request body
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # map model labels into pie chart buckets
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        # use fixed colors so chart looks same every time
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        # draw the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        # force equal axis so pie stays circular
        plt.axis('equal')

        # save chart into bytes in memory so flask can return an image
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # return the chart png to frontend
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500
    

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        # read comments from request
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # clean comments so cloud words are better
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # join all comments because wordcloud expects one text blob
        text = ' '.join(preprocessed_comments)

        # generate the word cloud image
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # keep image bytes in memory so we can send it directly
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # return wordcloud png to frontend
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500
    


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        # read sentiment timeline data from request
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # convert list data to dataframe for easy grouping
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # set timestamp as index to resample by month
        df.set_index('timestamp', inplace=True)

        # make sure sentiment values are integers for math
        df['sentiment'] = df['sentiment'].astype(int)

        # make readable labels for legend
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # count each sentiment for each month
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # get total comments in each month
        monthly_totals = monthly_counts.sum(axis=1)

        # convert counts to percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # make sure all 3 sentiment columns exist
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # keep sentiment columns in a fixed order
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # plot the trend lines
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # red for negative
            0: 'gray',     # gray for neutral
            1: 'green'     # green for positive
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # format dates so x-axis labels are easier to read
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # save trend graph into bytes so flask can return png
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # return trend graph image to frontend
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    


if __name__ == '__main__':
    # run api on port 5000 so extension can call it
    app.run(host='0.0.0.0', port=5000, debug=True)










