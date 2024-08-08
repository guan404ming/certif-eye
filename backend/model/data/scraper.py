import time
import pandas as pd
import os
from selenium import webdriver
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService


class GoogleMapScraper:
    def __init__(self, place_id, target_count=5):
        self.place_id = place_id
        self.target_count = target_count
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install())
        )
        self.driver.get(f"https://www.google.com/maps/place/?q=place_id:{place_id}")
        self.place_name = self.driver.find_element(By.CLASS_NAME, "DUwDvf").text

        WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//*[text()='評論']"))
        )

        self.driver.find_element(By.XPATH, "//*[text()='評論']").click()
        time.sleep(0.5)
        self.reviews = []

    def load_reviews(self):
        scroll_attempts = 0
        current_length = len(self.reviews)

        while current_length < self.target_count and scroll_attempts < 2:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "jftiEf"))
            )

            current_reviews = self.driver.find_elements(By.CLASS_NAME, "jftiEf")
            if current_reviews[-1]:
                ActionChains(self.driver).scroll_from_origin(
                    ScrollOrigin.from_element(current_reviews[-1]), 0, 10000
                ).perform()
                time.sleep(0.5)
                current_reviews = self.driver.find_elements(By.CLASS_NAME, "jftiEf")

            if len(current_reviews) == current_length:
                scroll_attempts += 1
            else:
                current_length = len(current_reviews)
                scroll_attempts = 0

        show_more_buttons = self.driver.find_elements(
            By.XPATH, "//*[contains(text(), '全文')]"
        )[: self.target_count]
        
        for show_more_button in show_more_buttons:
            ActionChains(self.driver).click(
                show_more_button
            ).perform()

        soup = Soup(self.driver.page_source, "lxml")
        self.reviews = soup.find_all(class_="jftiEf")[: self.target_count]
        print(f"Total reviews loaded: {len(self.reviews)}")

    def extract_reviews(self):
        def get_text(element, class_name):
            tag = element.find(class_=class_name)
            return tag.text if tag else ""

        def get_nickname_and_count(text):
            parts = text.split("·")
            return parts[0], parts[1].strip() if len(parts) > 1 else "1 review"

        data = []
        for review in self.reviews:
            name = get_text(review, "d4r55")
            nickname, review_count = get_nickname_and_count(get_text(review, "WNxzHc"))
            rating = (
                review.find("span", class_="kvMYJc")["aria-label"]
                if review.find("span", class_="kvMYJc")
                else ""
            )
            date = get_text(review, "rsqaWe")
            text = get_text(review, "wiI7pd")
            photo_urls = [
                btn["style"].split('url("')[1].split('")')[0]
                for btn in review.find_all("button", class_="Tya61d")
            ]
            photo = photo_urls[0] if photo_urls else ""
            data.append(
                [
                    self.place_id,
                    self.place_name,
                    name,
                    nickname,
                    review_count,
                    rating,
                    date,
                    text,
                    photo,
                ]
            )

        return pd.DataFrame(
            data,
            columns=[
                "place_id",
                "place_name",
                "reviewer",
                "nickname",
                "review_count",
                "rating",
                "date",
                "review",
                "review_photo",
            ],
        )

    def save_reviews_to_csv(self, filename="model/data/reviews.csv"):
        reviews_df = self.extract_reviews()
        if not os.path.isfile(filename):
            reviews_df.to_csv(filename, index=False, encoding="utf-8")
        else:
            reviews_df.to_csv(
                filename, mode="a", header=False, index=False, encoding="utf-8"
            )
        print(f"Reviews saved to {filename}")


# Usage example
# place_id = "ChIJ_fe8LwQ-aTQRPudaUDjIu8o"
# scraper = GoogleMapReviewScraper(place_id, target_count=10)
# scraper.load_reviews()
# scraper.save_reviews_to_csv()
