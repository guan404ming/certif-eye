import googlemaps
import time
import os
from dotenv import load_dotenv


class GoogleMapFinder:
    def __init__(self):
        load_dotenv()

        self.client = googlemaps.Client(key=os.getenv("API_KEY"))
        self.collected_restaurants = []

    def get_places(self, centers, type="restaurant", count=5):
        unique_restaurants = set(self.collected_restaurants)
        
        for center in centers:
            next_page_token = None

            while True:
                try:
                    response = self.client.places(
                        query="restaurant",
                        location=center,
                        radius=2000,
                        type=type,
                        page_token=(next_page_token if next_page_token else None),
                    )

                    restaurants = response.get("results", [])
                    next_page_token = response.get("next_page_token")

                    for restaurant in restaurants:
                        if restaurant["place_id"] not in unique_restaurants:
                            self.collected_restaurants.append(restaurant)
                        unique_restaurants.add(restaurant["place_id"])

                    if not next_page_token or len(unique_restaurants) >= count:
                        # if not next_page_token:
                        #     print("No more places to collect for center:", center)
                        # else:
                        #     print("Target count reached for center:", center)
                        break

                    time.sleep(2)

                except googlemaps.exceptions.ApiError as e:
                    print(f"API錯誤：{e}")
                    time.sleep(5)  # 等待一段時間後重試
                except Exception as e:
                    print(f"發生錯誤：{e}")
                    break

                if len(unique_restaurants) >= count:
                    # print("Target count reached for center:", center)
                    break

        return self.collected_restaurants[:count]


# usage
# finder = GoogleMapFinder()
# print(len(finder.find_place(count=100, centers=[[25.0129, 121.5371]])))
