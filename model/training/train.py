import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Define your neural network architecture
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

# Custom dataset class
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load your dataset
data = pd.read_csv("dataset.csv")
# Encode labels as 0 and 1
data["label"] = data["label"].apply(lambda x: 0 if x == "CG" else 1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data["review"], data["label"], test_size=0.5, random_state=42
)

# Convert text data into numerical vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
print("Data loaded and preprocessed")

# Define hyperparameters
input_size = X_train.shape[1]
hidden_size = 128
output_size = 1
learning_rate = 3e-4
num_epochs = 10
batch_size = 32

# Create DataLoader for batch processing
train_dataset = ReviewDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = BinaryClassifier(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == labels.unsqueeze(1)).float().mean()
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%')

# Evaluations
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y_test_tensor.unsqueeze(1)).float().mean()
    print(f"Accuracy: {accuracy.item()*100:.2f}%")

# Save the model and vectorizer
torch.save(model.state_dict(), "model.pth")
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
