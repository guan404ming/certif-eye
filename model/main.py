import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pickle

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
        self.model.load_state_dict(torch.load("model/lib/model.pth"))
        self.vectorizer = CountVectorizer()
        with open("model/lib/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        

    def infer(self, review):
        review_vector = self.vectorizer.transform([review]).toarray()
        review_tensor = torch.tensor(review_vector, dtype=torch.float32)
        output = self.model(review_tensor)
        return output.item()


model = Model()
print(model.infer("This is a great movie!"))