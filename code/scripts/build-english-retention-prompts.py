"""Bộ prompt chữ tiếng Anh để kiểm tra chống quên (RQ4): model sau fine-tune/RL có viết tiếng Anh kém đi không.

90 prompt (30 ngắn 1–3 từ / 30 vừa 4–10 từ / 30 dài 11–25 từ), caption tiếng Anh, chữ đích trong ngoặc kép,
seed cố định. Chấm như ViGlyph (score-baseline-from-ocr.py); với tiếng Anh, CER_NFD = CER_NFC.
Usage: python scripts/build-english-retention-prompts.py   → bench/english/prompts_en_v1.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261501
SHORT = ["Fresh Bakery", "OPEN", "Coffee Shop", "EXIT", "Book Store", "SALE", "Welcome Home", "Pizza", "Flower Market",
         "NO PARKING", "Thank You", "Daily News", "Ice Cream", "PHARMACY", "Good Morning", "Tea House", "Hotel",
         "Car Wash", "Happy Birthday", "Library", "Bus Stop", "FREE WIFI", "City Park", "Fish Market", "Taxi",
         "Garden Center", "Hair Salon", "Museum", "Night Club", "Bike Rental"]
MEDIUM = ["Fresh coffee and pastries every morning", "Please keep this door closed", "Grand opening this Saturday",
          "All items fifty percent off today", "Welcome to our little garden", "Handmade soap and natural candles",
          "Open daily from eight to six", "Reading is dreaming with open eyes", "Best noodles in the whole city",
          "Students get a free drink", "Mind the gap between train and platform", "Live music every Friday night",
          "Fresh fruit juice made to order", "Do not feed the animals", "The quick brown fox jumps",
          "Summer collection now in store", "Wash your hands before eating", "Turn left for the main hall",
          "Home made bread since 1985", "Quiet please, exam in progress", "Organic vegetables from local farms",
          "Lost and found at the front desk", "Order online and pick up here", "Keep calm and drink tea",
          "Annual science fair winners", "Back in five minutes", "Pet friendly cafe and bakery",
          "Fire exit keep clear at all times", "Happy new year to everyone", "Learn to code in twelve weeks"]
LONG = ["Thank you for visiting our museum, please remember to leave your feedback at the front desk",
        "Our shop will be closed for renovation next week and will reopen on Monday morning",
        "Every great journey begins with a single step and a little bit of courage",
        "Please switch off your mobile phone before the performance begins in the main hall",
        "The library is open from nine in the morning until eight in the evening every weekday",
        "Fresh seafood delivered daily from the harbor to your table with love and care",
        "Join us this weekend for a free cooking class with our head chef and his team",
        "Children under twelve must be accompanied by an adult at all times in the pool area",
        "The best way to predict the future is to create it with your own hands",
        "Visitors are kindly asked not to touch the paintings or take photos with flash",
        "Buy one coffee and get the second one free every Monday until the end of the month",
        "This trail is steep and slippery after rain, please wear proper shoes and stay on the path",
        "Our mission is to bring clean water and fresh food to every family in the village",
        "Registration for the spring semester closes on Friday, so please submit your forms early",
        "Happiness is not something ready made, it comes from your own actions every day",
        "Please wait here until a member of our staff calls your name and number",
        "The concert has been moved to the outdoor stage because of the beautiful weather today",
        "All proceeds from tonight's dinner will be donated to the local children's hospital",
        "Remember to bring your library card when you want to borrow more than three books",
        "The museum café serves breakfast, lunch and afternoon tea in a quiet garden setting",
        "Drive slowly through the school zone because children are crossing the road here",
        "We are proud to have served this neighborhood with fresh bread for over forty years",
        "Please recycle your bottles and cans in the blue bins located near the main entrance",
        "Our hotel offers free breakfast, free parking and a heated swimming pool for all guests",
        "Life is short, so eat the cake, read the book and call the people you love",
        "Training session starts at seven sharp, so please arrive ten minutes before the warm up",
        "The farmers market moves to the town square every Sunday morning during the summer",
        "Smoking is strictly prohibited inside the building and within ten meters of the entrance",
        "Learning a new language opens doors to new friends, new places and new ideas",
        "Please hold the handrail while using the escalator and keep children close to you"]
SIGN = ["A photo of a shop sign on an old city street with the words \"{t}\" in large clear letters.",
        "A flat modern poster with a pastel background and the text \"{t}\" in the center.",
        "A chalkboard in a cozy cafe with the words \"{t}\" handwritten in white chalk.",
        "A neon sign glowing on a brick wall at night that reads \"{t}\".",
        "A minimalist book cover with the bold title \"{t}\" in the middle.",
        "A product box on a white background with \"{t}\" printed on the front."]
LONG_T = ["A large poster on a wall with the paragraph \"{t}\" printed in a clean readable font.",
          "A sheet of white paper on a wooden desk with the text \"{t}\" printed in black ink.",
          "A notice board in a lobby with the message \"{t}\" printed on it."]


def main():
    rows = []
    for bucket, texts, tpls in (("short", SHORT, SIGN), ("medium", MEDIUM, SIGN), ("long", LONG, LONG_T)):
        for t in texts:
            i = len(rows)
            rows.append({"id": f"en1_{i:04d}", "text": t, "bucket": bucket, "n_words": len(t.split()),
                         "is_upper": t.isupper(), "prompt": tpls[i % len(tpls)].format(t=t), "seed": BASE_SEED + i})
    out = ROOT / "bench/english/prompts_en_v1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} prompts -> {out}")


if __name__ == "__main__":
    main()
