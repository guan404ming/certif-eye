from faker import Faker
import pandas as pd

# Load the CSV file and drop rows with NaN in 'review'
df = pd.read_csv("reviews.csv")
df.dropna(subset=["review"], inplace=True)

# Add 'label' column with value 'OR'
df["label"] = "OR"

# Get compound unique combinations of 'place_id' and 'place_name'
compound_unique = df[["place_id", "place_name"]].drop_duplicates()

# Initialize Faker with Traditional Chinese locale
fake = Faker("zh_TW")
num_fake_reviews = 10

# Generate fake reviews and append them to the DataFrame
fake_reviews = []

for place in compound_unique.values.tolist():
    print(f"Generating fake reviews for {place[1]}")
    for _ in range(num_fake_reviews):
        fake_review = {
            "place_id": place[0],
            "place_name": place[1],
            "reviewer": fake.name(),
            "nickname": f"{fake.name()}在地嚮導",
            "review_count": f"{fake.random_int(min=1, max=100)} 則評論",
            "rating": f"{fake.random_int(min=1, max=5)} 顆星",
            "date": fake.date_between(start_date="-1y", end_date="today").strftime("%Y-%m-%d"),
            "review": fake.text(max_nb_chars=200),
            "review_photo": "",
            "label": "CG"
        }
        fake_reviews.append(fake_review)

# Create a DataFrame for the fake reviews
df_fake_reviews = pd.DataFrame(fake_reviews)

# Concatenate the original DataFrame with the fake reviews DataFrame
df_combined = pd.concat([df, df_fake_reviews], ignore_index=True)

# Save the updated DataFrame to a CSV file
df_combined.to_csv("reviews_updated.csv", index=False)

# Display the first few rows of the updated DataFrame
df_combined.head(), df_combined.tail()
