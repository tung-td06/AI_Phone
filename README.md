# AI_Phone — Hệ Thống RAG Tư Vấn Điện Thoại Thông Minh (No LLM & Pipeline)

Hệ thống RAG tư vấn, tra cứu cấu hình và so sánh điện thoại di động thông minh, tối ưu hóa toàn diện từ Intent Detection, Query Understanding, Hybrid Retrieval (FAISS Dense + BM25 Sparse), Product Ranking (chấm điểm hiệu năng SoC chuẩn hóa, không hallucinate) và Evaluation Suite.

## 🚀 Các Giai Đoạn Phát Triển (Pipeline B1 -> B7)

- **`xuli.py`**: Tiền xử lý dữ liệu HTML PhoneDB thô sang Markdown sạch (`.md`), bảo toàn 100% dữ liệu kỹ thuật.
- **`B1.py`**: Đọc và nạp dữ liệu Markdown đa luồng (`PhoneDocument`).
- **`B2.py`**: Xây dựng kho dữ liệu Chunks & Metadata theo 8 chuyên mục kỹ thuật (Processor, Display, Camera, Battery, Connectivity, Memory, Dimensions, Thông tin chung).
- **`B3.py`**: Chuẩn bị tập dữ liệu Fine-tune (Positive Pairs & Triplets với Hard Negatives) chia theo thiết bị chống rò rỉ dữ liệu.
- **`B4.py`**: Fine-tune Sentence Transformers trên GPU (MNRL Loss, FP16 Mixed Precision).
- **`B5.py`**: Mã hóa toàn bộ dữ liệu thành Dense Vectors, xây dựng CSDL Vector (FAISS FlatIP) và BM25 Index.
- **`B6.py`**: Hệ thống Hybrid Search & RRF Fusion ban đầu.
- **`B7.py`**: **Nâng cấp Toàn diện (P0 -> P3)** — Chạy hoàn toàn độc lập, không cần LLM:
  - **P0 - Structured Query Understanding**: Bóc tách ngân sách thông minh (`BudgetParser`), nhận diện game (`GameTitleExtractor`), nhận diện hãng (`Brand Detection`), phân loại ý định (`IntentType`: RECOMMENDATION, COMPARISON, SPEC_LOOKUP, PRICE_QUERY, GENERAL_CHAT).
  - **P1 - Hybrid Retrieval Router**: Điều phối kích thước ứng viên và mở rộng truy vấn theo từng phân khúc ngân sách và miền nhu cầu.
  - **P2 - Dynamic Product Ranking & Price Parsing**: Parser giá đa tiền tệ (VND, USD, CNY, EUR), chấm điểm phần cứng (CPU, GPU, RAM, màn hình, tản nhiệt, pin) kết hợp bảng hiệu năng benchmark thực tế từ `soc_database.py`. Rào chắn trần cấu hình và loại bỏ thiết bị không phải điện thoại (Tablet/Smartwatch).
  - **P3 - Evaluation Suite**: 16 ca kiểm định toàn diện đo Recall@K, Precision@K, nDCG@K, MRR (đạt 100% 16/16 ca).

## ⚡ Triển Khai Trên Kaggle (End-to-End Pipeline)

### 1. Chuẩn bị trên Kaggle:
1. Tạo một Kaggle Notebook mới (File -> New Notebook).
2. Chọn **Accelerator**: **GPU T4 x2** hoặc **GPU P100** trong phần Settings.
3. Bấm **+ Add Input** (góc trên bên phải), tìm và gắn Dataset thô:
   - Link Dataset: [https://www.kaggle.com/datasets/tngtrn23/phone-db](https://www.kaggle.com/datasets/tngtrn23/phone-db)
   - Đường dẫn mount tự động: `/kaggle/input/phone-db`

### 2. Sử dụng Notebook Pipeline có sẵn:
Tải file [Kaggle_AI_Phone_Pipeline.ipynb](file:///p:/train%20AI/Kaggle_AI_Phone_Pipeline.ipynb) từ repo lên Kaggle (File -> Import Notebook) hoặc chạy tuần tự qua dòng lệnh:

```bash
# 1. Clone mã nguồn
cd /kaggle/working
git clone https://github.com/tung-td06/AI_Phone.git
cd /kaggle/working/AI_Phone

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Tiền xử lý dữ liệu thô HTML sang Markdown sạch
python xuli.py --input-dir /kaggle/input/phone-db --output-dir /kaggle/working/database_phone_cleaned --workers 8

# 4. Chunking thông số kỹ thuật (B2)
python B2.py --dir /kaggle/working/database_phone_cleaned --output /kaggle/working/chunks.jsonl

# 5. Sinh tập dữ liệu huấn luyện Triplets & Pairs (B3)
python B3.py --dir /kaggle/working/database_phone_cleaned --output-dir /kaggle/working/dataset_finetune --format both

# 6. Fine-tune mô hình Embedding trên GPU (B4)
python B4.py --data-dir /kaggle/working/dataset_finetune --output-dir /kaggle/working/final_phone_embedding_model --epochs 1 --batch-size 32

# 7. Xây dựng CSDL Vector FAISS & BM25 (B5)
python B5.py --dir /kaggle/working/database_phone_cleaned --model-dir /kaggle/working/final_phone_embedding_model --rag-db-dir /kaggle/working/rag_db --batch-size 128

# 8. Vận hành RAG & Đánh giá (B7)
python B7.py --rag-db-dir /kaggle/working/rag_db --model-dir /kaggle/working/final_phone_embedding_model --eval
```

## 🛠️ Cài Đặt Môi Trường Cục Bộ (Local)

```bash
pip install -r requirements.txt
```

## 💻 Hướng Dẫn Chạy B7 Cục Bộ

```bash
# 1. Truy vấn câu hỏi cụ thể:
python B7.py --query "tai chinh 15 trieu mua dien thoai nao" --top-k 5
python B7.py --query "May nao choi game tot nhat hien tai?" --top-k 5
python B7.py --query "Dien thoai choi PUBG tot duoi 10 trieu" --top-k 5

# 2. Chạy toàn bộ bộ kiểm thử 16 ca (Evaluation Suite):
python B7.py --eval

# 3. Giao diện REPL tương tác:
python B7.py
```

