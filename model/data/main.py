from model.lib.data.finder import GoogleMapFinder
from model.lib.data.scraper import GoogleMapScraper
import concurrent.futures


def main():
    finder = GoogleMapFinder()
    places = finder.get_places(
        count=500,
        centers=[
            [25.0129, 121.5371],  # NTU
            [25.0526, 121.4441],  # 中山
            [25.0330, 121.5654],  # 台北市中心
            [25.0353, 121.5611],  # 信義區
            [25.0478, 121.5318],  # 台北車站
            [25.0375, 121.5637],  # 忠孝東路
            [25.0828, 121.5664],  # 士林夜市
            [25.0328, 121.4996],  # 台北101
        ],
    )
    print(f"\nTotal places collected: {len(places)}")
    print("\n==============\n")

    def get_review(place):
        scraper = GoogleMapScraper(place, target_count=100)
        try:
            scraper.load_reviews()
            scraper.save_reviews_to_csv()
            print(f"{place['name']}: Total reviews extracted: {len(scraper.reviews)}")
            print("\n==============\n")
        except Exception as e:
            print(f"Error processing {place['name']}: {e}")
        finally:
            scraper.driver.quit()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_review, place) for place in places]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
