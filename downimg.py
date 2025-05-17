from icrawler.builtin import GoogleImageCrawler
import os

def download_images_icrawler(name, count=10, out_dir="dataset"):
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": folder})
    crawler.crawl(keyword=name + " face", max_num=count)

    print(f"\n✅ Images saved in: {folder}")

if __name__ == "__main__":
    person = input("👤 Enter person's name: ").strip()
    download_images_icrawler(person, count=30)
