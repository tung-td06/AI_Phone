"""
Module B2: Chia nhỏ (Chunking) dữ liệu thông số kỹ thuật điện thoại.
Thay vì embedding nguyên một file markdown dài, tài liệu được chia thành các đoạn nhỏ
theo từng chuyên mục kỹ thuật (Display, Processor, Memory, Camera, Battery, v.v.)
và bảo toàn đầy đủ metadata (brand, model, source_file, section).
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterator
from collections import defaultdict, Counter

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

from B1 import (
    DEFAULT_DATA_DIR,
    PhoneDocument,
    load_all_md_files,
    stream_md_files,
    parse_phone_metadata,
)

# Ánh xạ 22 danh mục gốc từ file markdown sang 8 nhóm logic theo yêu cầu
SECTION_MAPPING: Dict[str, str] = {
    # 1. Thông tin chung
    "Introduction": "Thông tin chung",
    "Software Environment": "Thông tin chung",
    "Geographical Attributes": "Thông tin chung",
    "Datasheet Attributes": "Thông tin chung",

    # 2. Dimensions (Kích thước & Thiết kế)
    "Physical Attributes": "Dimensions",
    "Control Peripherals": "Dimensions",
    "Ingress Protection": "Dimensions",

    # 3. Display (Màn hình)
    "Display": "Display",

    # 4. Processor/Performance (Vi xử lý & Hiệu năng)
    "Application processor, Chipset": "Processor/Performance",
    "Graphical Subsystem": "Processor/Performance",

    # 5. Memory (Bộ nhớ RAM & ROM)
    "Operative Memory": "Memory",
    "Non-volatile Memory": "Memory",

    # 6. Camera (Hệ thống máy ảnh Trước & Sau)
    "Primary Camera System": "Camera",
    "Secondary Camera System": "Camera",

    # 7. Battery (Pin & Nguồn điện)
    "Power Supply": "Battery",

    # 8. Connectivity (Kết nối, Mạng & Cảm biến)
    "Cellular Phone": "Connectivity",
    "Secondary Cellular Phone": "Connectivity",
    "Communication Interfaces": "Connectivity",
    "Multimedia Broadcast": "Connectivity",
    "Audio/Video Interfaces": "Connectivity",
    "Audio Subsystem": "Connectivity",
    "Built-in Sensors": "Connectivity",
}

# Thứ tự chuẩn hiển thị cho các section
CANONICAL_SECTIONS: List[str] = [
    "Thông tin chung",
    "Display",
    "Processor/Performance",
    "Memory",
    "Camera",
    "Battery",
    "Connectivity",
    "Dimensions",
]


@dataclass
class PhoneChunk:
    """Đại diện cho một đoạn chunk kèm metadata sẵn sàng cho Embedding/RAG."""
    chunk_id: str
    brand: str
    model: str
    source_file: str
    section: str
    content: str
    char_count: int
    token_estimate: int

    def to_dict(self) -> Dict:
        return asdict(self)


def extract_brand_and_model(content: str, filename: str) -> tuple[str, str]:
    """
    Trích xuất thương hiệu (brand) và model từ nội dung hoặc tên file.
    """
    brand = "Unknown"
    model = "Unknown"

    for line in content.splitlines()[:25]:
        line_clean = line.strip()
        if line_clean.startswith("Brand:"):
            brand = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Model:"):
            model = line_clean.split(":", 1)[1].strip()

    # Fallback nếu không parse được từ text
    if brand == "Unknown" or model == "Unknown":
        stem = Path(filename).stem
        parts = stem.split("_", 1)
        if brand == "Unknown" and parts:
            brand = parts[0].capitalize()
        if model == "Unknown":
            model = parts[1].replace("_", " ").title() if len(parts) > 1 else stem

    return brand, model


def slugify(text: str) -> str:
    """Chuyển chuỗi thành định dạng an toàn cho ID (slug)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "_", text).strip("_")


def chunk_phone_document(
    doc: PhoneDocument,
    mode: str = "grouped",
    add_context_header: bool = True
) -> List[PhoneChunk]:
    """
    Chia nhỏ một tài liệu điện thoại thành các chunk có cấu trúc.

    Args:
        doc: Đối tượng PhoneDocument
        mode:
          - "grouped": Gom 22 mục thành 8 nhóm chuyên đề chính (mặc định)
          - "raw": Tách riêng từng mục trong 22 mục gốc
        add_context_header: Thêm tiêu đề nhận diện máy vào đầu đoạn để embedding chính xác

    Returns:
        Danh sách PhoneChunk
    """
    brand, model = extract_brand_and_model(doc.content, doc.filename)
    file_stem = Path(doc.filename).stem

    # Tách tài liệu theo các header dạng [Tên mục]
    raw_sections = re.split(r"\n(?=\[[^\]]+\])", doc.content.strip())
    chunks: List[PhoneChunk] = []

    if mode == "grouped":
        grouped_sections = defaultdict(list)
        for raw in raw_sections:
            lines = raw.strip().splitlines()
            if not lines:
                continue
            match = re.match(r"^\[(.*?)\]", lines[0].strip())
            if match:
                sec_header = match.group(1).strip()
                group_name = SECTION_MAPPING.get(sec_header, "Thông tin chung")
                grouped_sections[group_name].append(raw.strip())
            else:
                grouped_sections["Thông tin chung"].append(raw.strip())

        # Tạo chunk theo thứ tự canonical nếu có
        ordered_groups = [g for g in CANONICAL_SECTIONS if g in grouped_sections]
        # Bổ sung các nhóm khác nếu phát sinh
        for g in grouped_sections:
            if g not in ordered_groups:
                ordered_groups.append(g)

        for sec_name in ordered_groups:
            body_text = "\n\n".join(grouped_sections[sec_name]).strip()
            if not body_text:
                continue

            if add_context_header:
                header = (
                    f"## Thiết bị: {brand} {model}\n"
                    f"- Hãng sản xuất: {brand}\n"
                    f"- Model: {model}\n"
                    f"- Chuyên mục: {sec_name}\n\n"
                )
                chunk_text = header + body_text
            else:
                chunk_text = body_text

            chunk_id = f"{file_stem}_{slugify(sec_name)}"
            chunks.append(
                PhoneChunk(
                    chunk_id=chunk_id,
                    brand=brand,
                    model=model,
                    source_file=doc.filename,
                    section=sec_name,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    token_estimate=len(chunk_text) // 4,
                )
            )

    else:  # mode == "raw"
        for idx, raw in enumerate(raw_sections):
            lines = raw.strip().splitlines()
            if not lines:
                continue
            match = re.match(r"^\[(.*?)\]", lines[0].strip())
            sec_header = match.group(1).strip() if match else f"Section_{idx+1}"

            if add_context_header:
                header = (
                    f"## Thiết bị: {brand} {model}\n"
                    f"- Hãng: {brand}\n"
                    f"- Model: {model}\n"
                    f"- Mục: {sec_header}\n\n"
                )
                chunk_text = header + raw.strip()
            else:
                chunk_text = raw.strip()

            chunk_id = f"{file_stem}_{slugify(sec_header)}"
            chunks.append(
                PhoneChunk(
                    chunk_id=chunk_id,
                    brand=brand,
                    model=model,
                    source_file=doc.filename,
                    section=sec_header,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    token_estimate=len(chunk_text) // 4,
                )
            )

    return chunks


def chunk_all_documents(
    documents: List[PhoneDocument],
    mode: str = "grouped",
    add_context_header: bool = True,
    show_progress: bool = True
) -> List[PhoneChunk]:
    """
    Chunk toàn bộ danh sách PhoneDocument.
    """
    all_chunks: List[PhoneChunk] = []
    use_tqdm = show_progress and (tqdm is not None)
    iterator = tqdm(documents, desc="Đang chunk dữ liệu", unit="doc") if use_tqdm else documents

    for doc in iterator:
        doc_chunks = chunk_phone_document(
            doc=doc,
            mode=mode,
            add_context_header=add_context_header
        )
        all_chunks.extend(doc_chunks)

    return all_chunks


def stream_chunks(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    limit: Optional[int] = None,
    mode: str = "grouped",
    add_context_header: bool = True
) -> Iterator[PhoneChunk]:
    """
    Generator đọc từng file từ ổ đĩa và sinh ra các chunk (tiết kiệm tối đa RAM).
    """
    for doc in stream_md_files(data_dir=data_dir, limit=limit):
        for chunk in chunk_phone_document(doc, mode=mode, add_context_header=add_context_header):
            yield chunk


def save_chunks_to_jsonl(chunks: List[PhoneChunk], output_file: str | Path) -> None:
    """Lưu danh sách chunks vào file JSON Lines (.jsonl)."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"[THÀNH CÔNG] Đã lưu {len(chunks):,} chunks vào: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="B2: Chunk dữ liệu điện thoại theo chuyên mục kỹ thuật")
    parser.add_argument("--dir", type=str, default=DEFAULT_DATA_DIR, help="Thư mục chứa file markdown")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số file đọc để test nhanh")
    parser.add_argument("--mode", type=str, default="grouped", choices=["grouped", "raw"],
                        help="Chế độ chunk: 'grouped' (8 nhóm chuẩn) hoặc 'raw' (từng section gốc)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file .jsonl xuất kết quả chunk")
    parser.add_argument("--no-header", action="store_true", help="Không gắn header ngữ cảnh (brand/model) vào đầu chunk")
    parser.add_argument("--no-progress", action="store_true", help="Tắt hiển thị thanh tiến trình")

    args = parser.parse_args()

    print("=" * 70)
    print(" BƯỚC 2: CHUNKING DỮ LIỆU ĐIỆN THOẠI CHO EMBEDDING / RAG")
    print("=" * 70)
    print(f"Thư mục nguồn      : {args.dir}")
    print(f"Chế độ chunk       : {args.mode.upper()} ({'8 nhóm chuyên đề' if args.mode == 'grouped' else '22 mục gốc'})")
    print(f"Gắn ngữ cảnh máy   : {'TẮT' if args.no_header else 'BẬT (Khuyến nghị cho Vector Search)'}")
    if args.limit:
        print(f"Giới hạn load      : {args.limit} file")
    print("-" * 70)

    # 1. Tải tài liệu qua B1
    load_start = time.time()
    docs = load_all_md_files(
        data_dir=args.dir,
        limit=args.limit,
        show_progress=not args.no_progress
    )
    load_elapsed = time.time() - load_start

    # 2. Thực hiện chunking
    chunk_start = time.time()
    chunks = chunk_all_documents(
        documents=docs,
        mode=args.mode,
        add_context_header=not args.no_header,
        show_progress=not args.no_progress
    )
    chunk_elapsed = time.time() - chunk_start
    total_elapsed = load_elapsed + chunk_elapsed

    # 3. Thống kê theo chuyên mục
    section_counter = Counter()
    section_char_sum = Counter()
    for c in chunks:
        section_counter[c.section] += 1
        section_char_sum[c.section] += c.char_count

    total_chars = sum(c.char_count for c in chunks)
    total_tokens = sum(c.token_estimate for c in chunks)

    print("-" * 70)
    print(" KẾT QUẢ CHUNKING:")
    print(f" - Tổng số điện thoại xử lý : {len(docs):,} máy")
    print(f" - Tổng số đoạn chunk tạo ra: {len(chunks):,} chunks")
    print(f" - Trung bình mỗi máy       : {len(chunks)/len(docs):.1f} chunks" if docs else "")
    print(f" - Tổng số ký tự            : {total_chars:,} chars (~{total_tokens:,} tokens ước tính)")
    print(f" - Thời gian tải file (B1)  : {load_elapsed:.2f}s")
    print(f" - Thời gian chunking (B2)  : {chunk_elapsed:.2f}s")
    print(f" - Tổng thời gian           : {total_elapsed:.2f}s")
    print("-" * 70)

    print(" CHI TIẾT THEO TỪNG CHUYÊN MỤC:")
    print(f" {'Section':<26} | {'Số lượng':<10} | {'Độ dài TB (chars)':<18} | {'Tokens TB':<10}")
    print(" " + "-" * 68)
    for sec, count in sorted(section_counter.items(), key=lambda x: x[0]):
        avg_char = section_char_sum[sec] / count if count else 0
        avg_tok = avg_char / 4
        print(f" {sec:<26} | {count:<10,} | {avg_char:<18.1f} | ~{avg_tok:<10.0f}")

    print("=" * 70)

    # 4. Xem thử 2 chunk mẫu đầu tiên
    if chunks:
        print("\n" + "=" * 70)
        print(" [XEM THỬ 2 CHUNK MẪU ĐẦU TIÊN]")
        print("=" * 70)
        for i, sample in enumerate(chunks[:2], 1):
            print(f"\n--- MẪU #{i} ---")
            print(f"Chunk ID    : {sample.chunk_id}")
            print(f"Brand       : {sample.brand}")
            print(f"Model       : {sample.model}")
            print(f"Source File : {sample.source_file}")
            print(f"Section     : {sample.section}")
            print(f"Tokens ước tính: ~{sample.token_estimate}")
            print("Nội dung:")
            lines = sample.content.strip().splitlines()
            for l in lines[:18]:
                print(f"  | {l}")
            if len(lines) > 18:
                print(f"  | ... ({len(lines) - 18} dòng nữa)")

    # 5. Lưu ra file nếu có chỉ định
    if args.output:
        save_chunks_to_jsonl(chunks, args.output)

    print("\nHoàn tất B2.py.")


if __name__ == "__main__":
    main()
