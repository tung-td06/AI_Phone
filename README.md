# AI_Phone — Hệ Thống RAG Tư Vấn Điện Thoại Thông Minh (No LLM & Pipeline)

Hệ thống RAG tư vấn, tra cứu cấu hình và so sánh điện thoại di động thông minh, tối ưu hóa toàn diện từ Intent Detection, Query Understanding, Hybrid Retrieval (FAISS Dense + BM25 Sparse), Product Ranking (chấm điểm hiệu năng SoC chuẩn hóa, không hallucinate) và Evaluation Suite.

## 🚀 Các Giai Đoạn Phát Triển (Pipeline B1 -> B7)

- **`B1.py`**: Chuẩn hóa dữ liệu thô, tiền xử lý thông số kỹ thuật điện thoại.
- **`B2.py`**: Xây dựng kho dữ liệu Chunks & Metadata đa chuyên mục (Processor, Display, Camera, Battery, Connectivity...).
- **`B3.py`**: Fine-tune Sentence Transformers trên tập dữ liệu domain điện thoại.
- **`B4.py`**: Đánh giá Embedding Model sau fine-tune.
- **`B5.py`**: Xây dựng CSDL Vector (FAISS) và BM25 Index.
- **`B6.py`**: Hệ thống Hybrid Search & RRF Fusion ban đầu.
- **`B7.py`**: **Nâng cấp Toàn diện (P0 -> P3)** — Chạy hoàn toàn độc lập, không cần LLM:
  - **P0 - Structured Query Understanding**: Bóc tách ngân sách thông minh (`BudgetParser`), nhận diện game (`GameTitleExtractor`), nhận diện hãng (`Brand Detection`), phân loại ý định (`IntentType`: RECOMMENDATION, COMPARISON, SPEC_LOOKUP, PRICE_QUERY, GENERAL_CHAT).
  - **P1 - Hybrid Retrieval Router**: Điều phối kích thước ứng viên và mở rộng truy vấn theo từng phân khúc ngân sách và miền nhu cầu.
  - **P2 - Dynamic Product Ranking & Price Parsing**: Parser giá đa tiền tệ (VND, USD, CNY, EUR), chấm điểm phần cứng (CPU, GPU, RAM, màn hình, tản nhiệt, pin) kết hợp bảng hiệu năng benchmark thực tế từ `soc_database.py`. Rào chắn trần cấu hình và loại bỏ thiết bị không phải điện thoại (Tablet/Smartwatch).
  - **P3 - Evaluation Suite**: 16 ca kiểm định toàn diện đo Recall@K, Precision@K, nDCG@K, MRR (đạt 100% 16/16 ca).

## 🛠️ Cài Đặt Môi Trường

```bash
pip install -r requirements.txt
# hoặc
pip install torch sentence-transformers faiss-cpu rank-bm25 numpy
```

## 💻 Hướng Dẫn Chạy B7

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
