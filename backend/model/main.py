import base64
import cv2
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
import googletrans
from wordcloud import WordCloud, STOPWORDS

from model.data.scraper import GoogleMapScraper
from flask import jsonify

input_size = 36975
hidden_size = 128
output_size = 1


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Most frequent words in reviews, ignoring specific words
stopwords = set(STOPWORDS)
stopwords.update(["restaurant", "place", "但", "不過"])


def get_place_rank(place_id):
    df_place = pd.read_csv("model/data/places.csv")
    # Rank the scores
    df_place["rank"] = df_place["score"].rank(ascending=True, method="min")

    # Calculate the rank percentage
    df_place["rank_percentage"] = (df_place["rank"] - 1) / (len(df_place) - 1) * 100

    # Find the rank percentage of the current score
    current_rank_percentage = df_place.loc[
        df_place["place_id"] == place_id, "rank_percentage"
    ].values[0]

    return current_rank_percentage


def get_place_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        font_path="model/font.ttf",
    ).generate(text)

    # Convert the word cloud to an image array
    wordcloud_array = wordcloud.to_array()

    # Convert the image array to a format suitable for OpenCV
    wordcloud_image = cv2.cvtColor(wordcloud_array, cv2.COLOR_RGB2BGR)

    # Encode the image as a PNG
    _, buffer = cv2.imencode(".png", wordcloud_image)

    # Encode the buffer to base64
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return img_base64


class Model:
    def __init__(self):
        self.model = BinaryClassifier(input_size, hidden_size, output_size)
        self.model.load_state_dict(torch.load("model/model.pth"))
        self.vectorizer = CountVectorizer()
        with open("model/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def infer(self, review):
        review_vector = self.vectorizer.transform([review]).toarray()
        review_tensor = torch.tensor(review_vector, dtype=torch.float32)
        output = self.model(review_tensor)
        return output.item()

    def get_place_info(self, place_id):
        # Read the places.csv file
        df_place = pd.read_csv("model/data/places.csv")
        df_reviews = pd.read_csv("model/data/reviews.csv")
        df_reviews = df_reviews[df_reviews["place_id"] == place_id]

        # Check if the place_id exists in places.csv
        if place_id in df_place["place_id"].values:
            df_place_row = df_place[df_place["place_id"] == place_id]
            if not pd.isna(df_place_row["score"].values[0]):
                text = " ".join(df_reviews["review"])
                return jsonify(
                    {
                        "score": get_place_rank(place_id),
                        "wordcloud": get_place_wordcloud(text),
                    }
                )

        # Check if there are any reviews for the place_id
        if len(df_reviews) == 0:
            scraper = GoogleMapScraper(place_id, target_count=30)
            scraper.load_reviews()
            scraper.save_reviews_to_csv(filename="model/data/reviews.csv")
            df_reviews = pd.read_csv("model/data/reviews.csv")
            df_reviews = df_reviews[df_reviews["place_id"] == place_id]
            if len(scraper.reviews) == 0:
                return -1

        # Calculate the total score from reviews
        total = 0
        translator = googletrans.Translator()
        for review in df_reviews["review"]:
            translated_text = translator.translate(str(review), dest="en").text
            total += self.infer(translated_text)
        translator.client.close()

        score = total / len(df_reviews)

        # Append or update the score for the place_id in places.csv
        if place_id in df_place["place_id"].values:
            df_place.loc[df_place["place_id"] == place_id, "score"] = score
        else:
            new_row = pd.DataFrame({"place_id": [place_id], "score": [score]})
            df_place = pd.concat([df_place, new_row], ignore_index=True)

        df_place.to_csv("model/data/places.csv", index=False)

        return jsonify(
            {
                "score": get_place_rank(place_id),
                "wordcloud": get_place_wordcloud(" ".join(df_reviews["review"])),
            }
        )
