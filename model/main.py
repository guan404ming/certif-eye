import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
import googletrans

from model.data.scraper import GoogleMapScraper

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

    def get_place_score(self, place_id):
        # Read the places.csv file
        df_place = pd.read_csv("model/data/places.csv")

        # Check if the place_id exists in places.csv
        if place_id in df_place["place_id"].values:
            df_place_row = df_place[df_place["place_id"] == place_id]
            if not pd.isna(df_place_row["score"].values[0]):
                return df_place_row["score"].values[0]

        # Read the reviews.csv file
        df_reviews = pd.read_csv("model/data/reviews.csv")
        df_reviews = df_reviews[df_reviews["place_id"] == place_id]

        # Check if there are any reviews for the place_id
        if len(df_reviews) == 0:
            scraper = GoogleMapScraper(place_id, target_count=30)
            scraper.load_reviews()
            scraper.save_reviews_to_csv(filename="model/data/reviews.csv")
            df_reviews = pd.read_csv("model/data/reviews.csv")
            df_reviews = df_reviews[df_reviews["place_id"] == place_id]
            if (len(scraper.reviews) == 0):
                return -100

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

        return score
