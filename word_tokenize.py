import pandas as pd
from underthesea import word_tokenize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Hàm xử lý cho 1 item: input = (index, text) -> trả về (index, tokenized_text)
def tokenize_item(item):
    idx, text = item
    tokens = word_tokenize(str(text), format="text")  # ép về str để tránh lỗi NaN
    return idx, tokens

if __name__ == "__main__":
    start = time.time()

    # 1) Đọc CSV gốc
    df = pd.read_csv("Cleaned_news_dataset.csv", encoding="utf-8")   # hoặc "utf-8-sig"
    n = len(df)
    print(f"📊 Tổng {n} dòng cần xử lý.")

    # 2) Số core để dùng (chừa 2 core cho hệ điều hành)
    num_cores = max(1, cpu_count() - 1)
    print(f"⚡ Dùng {num_cores}/{cpu_count()} core để xử lý song song...")

    # 3) Chuẩn bị dữ liệu (index, text)
    items = list(enumerate(df["clean_text"].astype(str)))

    # 4) Mảng kết quả tạm
    results = [None] * n

    # 5) chunksize tự động (toàn dataset chia đều cho số core * 10)
    chunksize = max(1, n // (num_cores * 10))
    print(f"⚙️ chunksize = {chunksize}")

    # 6) Tokenize song song với progress bar
    with Pool(processes=num_cores) as pool:
        for idx, tokenized in tqdm(pool.imap_unordered(tokenize_item, items, chunksize=chunksize), total=n):
            results[idx] = tokenized

    # 7) Gắn kết quả vào DataFrame
    df["tokens"] = results
    df = df.drop(columns=["clean_text"])  # bỏ cột gốc nếu không cần
    # 8) Xuất ra CSV (utf-8-sig để Excel đọc tiếng Việt không lỗi)
    df.to_csv("Data_tokenized.csv", index=False, encoding="utf-8-sig")

    print(f"✅ Xong! File CSV lưu tại: data_tokenized.csv")
    print(f"⏱️ Tổng thời gian: {time.time() - start:.2f} giây")