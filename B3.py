"""
Module B3: Chuẩn bị dữ liệu Fine-tune cho SentenceTransformers / Embedding Models.

Chuyển đổi các chunk thông số kỹ thuật điện thoại (từ B2) thành:
1. Cặp tích cực (Positive Pairs): (Query, Positive Passage) dùng cho MultipleNegativesRankingLoss (MNRL).
2. Bộ ba tương phản (Triplets): (Anchor, Positive, Hard Negative) dùng cho TripletLoss.
   - Hard Negative 1: Cùng máy nhưng khác chuyên mục (chống nhầm lẫn tính năng).
   - Hard Negative 2: Cùng chuyên mục nhưng khác máy (chống nhầm lẫn model).

Bảo toàn đầy đủ metadata (brand, model, source_file, section) và phân chia Train/Validation
theo từng thiết bị (tránh data leakage).
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict

# Đảm bảo terminal Windows in được tiếng Việt Unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from B1 import DEFAULT_DATA_DIR, load_all_md_files
from B2 import PhoneChunk, chunk_all_documents, CANONICAL_SECTIONS


# ==============================================================================
# 1. TEMPLATES CÂU HỎI TỰ NHIÊN THEO TỪNG CHUYÊN MỤC
# ==============================================================================
QUERY_TEMPLATES: Dict[str, List[str]] = {
    "Battery": [
        "pin {phone} thế nào?",
        "dung lượng pin của {phone} là bao nhiêu?",
        "{phone} pin bao nhiêu mAh?",
        "thời lượng pin và sạc của {phone}",
        "thông số pin {phone}",
        "{phone} pin dùng được bao lâu?",
        "{phone} có pin trâu không?",
        "thời gian đàm thoại của {phone}",
        "{phone} battery capacity and charging specs",
        "pin điện thoại {phone}",
    ],
    "Processor/Performance": [
        "{phone} dùng chip gì?",
        "vi xử lý và hiệu năng của {phone}",
        "cấu hình CPU GPU {phone}",
        "{phone} chạy chip gì, xung nhịp bao nhiêu?",
        "thông số chip xử lý {phone}",
        "{phone} processor and chipset specs",
        "con chip trên {phone} có mạnh không?",
        "card đồ họa GPU của {phone} là gì?",
        "hiệu năng chơi game {phone}",
    ],
    "Camera": [
        "camera {phone} chụp hình thế nào?",
        "thông số camera sau và camera trước của {phone}",
        "{phone} camera bao nhiêu megapixel?",
        "khả năng quay video và cảm biến camera {phone}",
        "đánh giá camera {phone}",
        "camera {phone} có những tính năng gì?",
        "{phone} camera specifications",
        "hệ thống camera của {phone} có flash và chống rung không?",
        "chất lượng chụp ảnh của {phone}",
    ],
    "Display": [
        "màn hình {phone} có kích thước bao nhiêu inch?",
        "độ phân giải và tấm nền màn hình {phone}",
        "{phone} dùng màn hình gì, độ phân giải thế nào?",
        "thông số màn hình hiển thị {phone}",
        "{phone} display resolution and screen size",
        "màn hình {phone} có kính cường lực chống trầy không?",
        "mật độ điểm ảnh PPI của màn hình {phone}",
        "chất lượng hiển thị của {phone}",
    ],
    "Memory": [
        "{phone} có bao nhiêu GB RAM và bộ nhớ trong?",
        "dung lượng RAM và ROM của {phone}",
        "{phone} có hỗ trợ thẻ nhớ ngoài không?",
        "bộ nhớ {phone} gồm những phiên bản nào?",
        "{phone} RAM and internal storage capacity",
        "{phone} dung lượng bộ nhớ khả dụng là bao nhiêu?",
        "loại RAM trên {phone} là gì?",
        "khả năng mở rộng bộ nhớ của {phone}",
    ],
    "Dimensions": [
        "kích thước và trọng lượng của {phone}",
        "{phone} nặng bao nhiêu gram, dày bao nhiêu mm?",
        "thiết kế kích thước vật lý {phone}",
        "{phone} dimensions and weight",
        "{phone} cầm có nặng tay không?",
        "{phone} có chống nước chống bụi không?",
        "độ mỏng và chiều dài rộng của {phone}",
    ],
    "Connectivity": [
        "{phone} có hỗ trợ 5G không?",
        "kết nối sim, wifi, bluetooth và cổng sạc của {phone}",
        "{phone} dùng cổng kết nối gì, có jack 3.5mm không?",
        "các băng tần mạng và kết nối của {phone}",
        "{phone} dùng mấy SIM, kích cỡ SIM gì?",
        "{phone} connectivity wifi bluetooth 5G and SIM",
        "{phone} có cổng tai nghe 3.5mm và GPS không?",
    ],
    "Thông tin chung": [
        "thông tin chung về điện thoại {phone}",
        "{phone} ra mắt năm nào, chạy hệ điều hành gì?",
        "tổng quan cấu hình {phone}",
        "{phone} specs and release date",
        "{phone} thuộc phân khúc nào, do ai sản xuất?",
        "hệ điều hành gốc của {phone} là gì?",
        "giới thiệu tổng quan về máy {phone}",
    ],
}


# ==============================================================================
# 2. CẤU TRÚC DỮ LIỆU CẶP (PAIR) VÀ BỘ BA (TRIPLET)
# ==============================================================================
@dataclass
class TrainPair:
    """Cặp (Query, Positive Passage) dùng cho MultipleNegativesRankingLoss."""
    query: str
    positive: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainTriplet:
    """Bộ ba (Anchor, Positive, Negative) dùng cho TripletLoss / Hard Negative Mining."""
    anchor: str
    positive: str
    negative: str
    negative_type: str  # 'same_phone_diff_section' hoặc 'diff_phone_same_section'
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


def get_natural_phone_name(brand: str, model: str) -> str:
    """Tạo tên điện thoại tự nhiên, tránh lặp lại tên hãng."""
    brand = brand.strip()
    model = model.strip()
    if model.lower().startswith(brand.lower()):
        return model
    return f"{brand} {model}"


def generate_queries_for_chunk(chunk: PhoneChunk, num_queries: int = 2) -> List[str]:
    """
    Tạo danh sách các câu hỏi tự nhiên tương ứng với chuyên mục của chunk.
    """
    phone_name = get_natural_phone_name(chunk.brand, chunk.model)
    templates = QUERY_TEMPLATES.get(chunk.section, QUERY_TEMPLATES["Thông tin chung"])

    selected_templates = random.sample(templates, min(num_queries, len(templates)))
    queries = [tpl.format(phone=phone_name) for tpl in selected_templates]
    return queries


# ==============================================================================
# 3. TẠO TẬP DỮ LIỆU POSITIVE PAIRS & TRIPLETS
# ==============================================================================
def build_positive_pairs(
    chunks: List[PhoneChunk],
    queries_per_chunk: int = 2
) -> List[TrainPair]:
    """
    Xây dựng tập (Query, Positive Passage) cho toàn bộ danh sách chunks.
    """
    pairs: List[TrainPair] = []
    for chunk in chunks:
        queries = generate_queries_for_chunk(chunk, num_queries=queries_per_chunk)
        for q in queries:
            pair = TrainPair(
                query=q,
                positive=chunk.content,
                metadata={
                    "brand": chunk.brand,
                    "model": chunk.model,
                    "source_file": chunk.source_file,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                }
            )
            pairs.append(pair)
    return pairs


def build_triplets(
    chunks: List[PhoneChunk],
    queries_per_chunk: int = 2,
    seed: int = 42
) -> List[TrainTriplet]:
    """
    Xây dựng tập (Anchor, Positive, Negative) với Hard Negatives được khai thác từ:
    - Loại 1 (50%): Cùng máy nhưng khác chuyên mục (buộc mô hình học đúng tính năng hỏi).
    - Loại 2 (50%): Cùng chuyên mục nhưng thuộc máy khác của cùng hãng (buộc mô hình học đúng model).
    """
    rng = random.Random(seed)

    # 1. Tạo chỉ mục tìm kiếm nhanh
    # chunks_by_phone[source_file][section] = chunk
    chunks_by_phone: Dict[str, Dict[str, PhoneChunk]] = defaultdict(dict)
    # chunks_by_brand_section[brand][section] = [chunk, ...]
    chunks_by_brand_section: Dict[str, Dict[str, List[PhoneChunk]]] = defaultdict(lambda: defaultdict(list))
    # all_chunks_by_section[section] = [chunk, ...]
    all_chunks_by_section: Dict[str, List[PhoneChunk]] = defaultdict(list)

    for c in chunks:
        chunks_by_phone[c.source_file][c.section] = c
        chunks_by_brand_section[c.brand][c.section].append(c)
        all_chunks_by_section[c.section].append(c)

    triplets: List[TrainTriplet] = []

    for chunk in chunks:
        queries = generate_queries_for_chunk(chunk, num_queries=queries_per_chunk)

        # Chuẩn bị danh sách ứng viên hard negative loại 1 (cùng máy, khác section)
        other_sections_same_phone = [
            other_c for sec, other_c in chunks_by_phone[chunk.source_file].items()
            if sec != chunk.section
        ]

        # Chuẩn bị danh sách ứng viên hard negative loại 2 (khác máy, cùng section)
        other_phones_same_brand_section = [
            other_c for other_c in chunks_by_brand_section[chunk.brand][chunk.section]
            if other_c.source_file != chunk.source_file
        ]
        # Nếu hãng đó chỉ có 1 máy, lấy từ các máy khác của hãng khác cùng section
        fallback_other_phones = [
            other_c for other_c in all_chunks_by_section[chunk.section]
            if other_c.source_file != chunk.source_file
        ]

        for idx, q in enumerate(queries):
            # Luân phiên chọn loại 1 và loại 2
            choose_type_1 = (idx % 2 == 0) and bool(other_sections_same_phone)

            if choose_type_1 and other_sections_same_phone:
                neg_chunk = rng.choice(other_sections_same_phone)
                neg_type = "same_phone_diff_section"
            elif other_phones_same_brand_section:
                neg_chunk = rng.choice(other_phones_same_brand_section)
                neg_type = "diff_phone_same_section"
            elif fallback_other_phones:
                neg_chunk = rng.choice(fallback_other_phones)
                neg_type = "diff_phone_same_section"
            elif other_sections_same_phone:
                neg_chunk = rng.choice(other_sections_same_phone)
                neg_type = "same_phone_diff_section"
            else:
                continue

            triplet = TrainTriplet(
                anchor=q,
                positive=chunk.content,
                negative=neg_chunk.content,
                negative_type=neg_type,
                metadata={
                    "brand": chunk.brand,
                    "model": chunk.model,
                    "source_file": chunk.source_file,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                    "neg_source_file": neg_chunk.source_file,
                    "neg_section": neg_chunk.section,
                }
            )
            triplets.append(triplet)

    return triplets


# ==============================================================================
# 4. CHIA TẬP TRAIN / VALIDATION THEO THIẾT BỊ (TRÁNH DATA LEAKAGE)
# ==============================================================================
def split_by_device(
    items: List[TrainPair | TrainTriplet],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List, List]:
    """
    Phân chia train và val dựa trên thiết bị (source_file).
    Điều này đảm bảo các máy trong tập validation là máy HOÀN TOÀN MỚI
    mà mô hình chưa từng thấy trong tập train.
    """
    rng = random.Random(seed)
    # Lấy toàn bộ danh sách các file thiết bị duy nhất
    unique_devices = list({it.metadata["source_file"] for it in items})
    rng.shuffle(unique_devices)

    split_point = int(len(unique_devices) * train_ratio)
    train_devices = set(unique_devices[:split_point])
    val_devices = set(unique_devices[split_point:])

    train_set = [it for it in items if it.metadata["source_file"] in train_devices]
    val_set = [it for it in items if it.metadata["source_file"] in val_devices]

    return train_set, val_set


def save_jsonl(items: List[TrainPair | TrainTriplet], output_path: str | Path) -> None:
    """Lưu danh sách cặp/bộ ba ra file JSON Lines (.jsonl)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    print(f"[ĐÃ LƯU] {len(items):,} mẫu vào: {out_file.resolve()}")


def generate_training_script_example(output_dir: Path) -> None:
    """
    Tạo file run_train_sentence_transformer.py mẫu để người dùng có thể
    chạy train ngay lập tức bằng SentenceTransformers.
    """
    script_content = '''"""
Kịch bản huấn luyện SentenceTransformer mẫu với dữ liệu tạo từ B3.py.
Yêu cầu cài đặt:
    pip install torch sentence-transformers datasets
"""

import json
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)

def load_jsonl_dataset(file_path):
    queries, positives = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])
            positives.append(data["positive"])
    return Dataset.from_dict({"anchor": queries, "positive": positives})

def main():
    print("1. Tải dữ liệu huấn luyện...")
    train_dataset = load_jsonl_dataset("train_pairs.jsonl")
    val_dataset = load_jsonl_dataset("val_pairs.jsonl")
    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    print("2. Khởi tạo mô hình nền (Embedding base model)...")
    # Có thể chọn: "BAAI/bge-small-en-v1.5", "BAAI/bge-m3", "intfloat/multilingual-e5-small"
    model_name = "BAAI/bge-m3"
    model = SentenceTransformer(model_name)

    print("3. Cấu hình Loss: MultipleNegativesRankingLoss...")
    loss = losses.MultipleNegativesRankingLoss(model)

    print("4. Cấu hình tham số huấn luyện...")
    args = SentenceTransformerTrainingArguments(
        output_dir="checkpoint_phone_embedding",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        fp16=True,  # Bật nếu có GPU NVIDIA hỗ trợ FP16
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    print("5. Bắt đầu Fine-tuning...")
    trainer.train()

    print("6. Lưu mô hình hoàn thiện...")
    model.save_pretrained("final_phone_embedding_model")
    print("Đã hoàn tất!")

if __name__ == "__main__":
    main()
'''
    train_script_path = output_dir / "run_train_sentence_transformer.py"
    with open(train_script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"[ĐÃ TẠO] Script huấn luyện mẫu: {train_script_path.resolve()}")


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FINETUNE_DIR = str(BASE_DIR / "dataset_finetune")


# ==============================================================================
# 5. HÀM CHÍNH VÀ GIAO DIỆN CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="B3: Chuẩn bị dữ liệu Fine-tune cho SentenceTransformers")
    parser.add_argument("--dir", type=str, default=DEFAULT_DATA_DIR, help="Thư mục chứa file markdown")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số thiết bị đọc để test nhanh (mặc định: toàn bộ)")
    parser.add_argument("--queries-per-chunk", type=int, default=2, help="Số câu hỏi sinh ra cho mỗi chunk (mặc định: 2)")
    parser.add_argument("--format", type=str, default="both", choices=["pairs", "triplets", "both"],
                        help="Định dạng dữ liệu: 'pairs' (Query-Positive), 'triplets' (Anchor-Pos-Neg), hoặc 'both'")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_FINETUNE_DIR,
                        help=f"Thư mục lưu dataset xuất ra (mặc định: {DEFAULT_FINETUNE_DIR})")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Tỷ lệ tập Train (mặc định: 0.9 = 90% train, 10% val)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để tái lập kết quả")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 75)
    print(" BƯỚC 3: CHUẨN BỊ DỮ LIỆU FINE-TUNE CHO SENTENCETRANSFORMER")
    print("=" * 75)
    print(f"Thư mục nguồn      : {args.dir}")
    print(f"Số câu hỏi / chunk : {args.queries_per_chunk}")
    print(f"Định dạng xuất     : {args.format.upper()}")
    print(f"Tỷ lệ Train / Val  : {int(args.train_ratio*100)}% / {int((1-args.train_ratio)*100)}% (Chia theo thiết bị)")
    print(f"Thư mục đầu ra     : {args.output_dir}")
    if args.limit:
        print(f"Giới hạn load      : {args.limit} máy")
    print("-" * 75)

    # 1. Tải và chunk dữ liệu qua B1 & B2
    t0 = time.time()
    docs = load_all_md_files(data_dir=args.dir, limit=args.limit)
    chunks = chunk_all_documents(documents=docs, mode="grouped", add_context_header=True)
    t_chunk = time.time() - t0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Xử lý Positive Pairs (nếu format là pairs hoặc both)
    if args.format in ["pairs", "both"]:
        print("\n[1/2] Đang tạo Positive Pairs (Query - Positive Passage)...")
        pairs = build_positive_pairs(chunks, queries_per_chunk=args.queries_per_chunk)
        train_pairs, val_pairs = split_by_device(pairs, train_ratio=args.train_ratio, seed=args.seed)

        save_jsonl(train_pairs, out_dir / "train_pairs.jsonl")
        save_jsonl(val_pairs, out_dir / "val_pairs.jsonl")

        print(f" -> Tổng Pairs: {len(pairs):,} | Train: {len(train_pairs):,} | Val: {len(val_pairs):,}")

    # 3. Xử lý Triplets với Hard Negatives (nếu format là triplets hoặc both)
    if args.format in ["triplets", "both"]:
        print("\n[2/2] Đang khai thác Hard Negatives và tạo Triplets (Anchor - Pos - Neg)...")
        triplets = build_triplets(chunks, queries_per_chunk=args.queries_per_chunk, seed=args.seed)
        train_triplets, val_triplets = split_by_device(triplets, train_ratio=args.train_ratio, seed=args.seed)

        save_jsonl(train_triplets, out_dir / "train_triplets.jsonl")
        save_jsonl(val_triplets, out_dir / "val_triplets.jsonl")

        print(f" -> Tổng Triplets: {len(triplets):,} | Train: {len(train_triplets):,} | Val: {len(val_triplets):,}")

    # 4. Sinh file kịch bản mẫu huấn luyện
    generate_training_script_example(out_dir)

    total_time = time.time() - t0

    print("-" * 75)
    print(" TỔNG KẾT BƯỚC 3:")
    print(f" - Tổng thiết bị xử lý  : {len(docs):,} máy")
    print(f" - Tổng chunk cơ sở     : {len(chunks):,} chunks")
    print(f" - Thời gian thực thi   : {total_time:.2f} giây")
    print(f" - Vị trí lưu trữ       : {out_dir.resolve()}")
    print("=" * 75)

    # 5. Xem thử 2 mẫu Pair và 1 mẫu Triplet
    if args.format in ["pairs", "both"] and pairs:
        print("\n[XEM THỬ 2 MẪU PAIR ĐẦU TIÊN]:")
        for i, p in enumerate(pairs[:2], 1):
            print(f"\n--- PAIR #{i} ---")
            print(f"Query    : \"{p.query}\"")
            print(f"Metadata : {p.metadata}")
            print(f"Positive : {p.positive.splitlines()[:6]}")
            print("  | ...")

    if args.format in ["triplets", "both"] and triplets:
        print("\n[XEM THỬ 1 MẪU TRIPLET VỚI HARD NEGATIVE]:")
        sample_trip = triplets[0]
        print(f"Anchor        : \"{sample_trip.anchor}\"")
        print(f"Negative Type : {sample_trip.negative_type}")
        print(f"Positive Chuyên mục: {sample_trip.metadata['section']} ({sample_trip.metadata['model']})")
        print(f"Negative Chuyên mục: {sample_trip.metadata['neg_section']} ({sample_trip.metadata['neg_source_file']})")

    print("\nHoàn tất B3.py.")


if __name__ == "__main__":
    main()
