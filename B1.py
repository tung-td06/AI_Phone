"""
Module B1: Load và đọc toàn bộ các file Markdown (.md) từ cơ sở dữ liệu điện thoại.
Mặc định: P:\\database phone cleaned
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def find_default_data_dir() -> str:
    kaggle_paths = [
        "/kaggle/working/database_phone_cleaned",
        "/kaggle/input/database-phone-cleaned",
        "/kaggle/input/phone-db-cleaned",
    ]
    for p in kaggle_paths:
        if os.path.exists(p):
            return p
    if os.path.exists(r"P:\database phone cleaned"):
        return r"P:\database phone cleaned"
    if os.path.exists("database_phone_cleaned"):
        return "database_phone_cleaned"
    return "/kaggle/working/database_phone_cleaned" if os.path.exists("/kaggle") else r"P:\database phone cleaned"

DEFAULT_DATA_DIR = find_default_data_dir()


@dataclass
class PhoneDocument:
    """Đại diện cho một tài liệu cấu hình điện thoại từ file .md."""
    filename: str
    filepath: str
    content: str
    size_bytes: int
    char_count: int
    line_count: int

    def to_dict(self) -> Dict:
        return asdict(self)


def get_md_file_paths(data_dir: str | Path, pattern: str = "*.md") -> List[Path]:
    """
    Quét và lấy toàn bộ danh sách đường dẫn các file .md trong thư mục.
    Hỗ trợ cả quét trực tiếp và đệ quy qua các thư mục con.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {path.resolve()}")
    if not path.is_dir():
        raise NotADirectoryError(f"Đường dẫn không phải là thư mục: {path.resolve()}")

    files = list(path.glob(pattern))
    if not files:
        files = list(path.rglob(pattern))
    return sorted(files)


def read_single_md_file(file_path: Path, encoding: str = "utf-8") -> Optional[PhoneDocument]:
    """
    Đọc một file .md đơn lẻ và trả về PhoneDocument.
    """
    try:
        size = file_path.stat().st_size
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        return PhoneDocument(
            filename=file_path.name,
            filepath=str(file_path.resolve()),
            content=content,
            size_bytes=size,
            char_count=len(content),
            line_count=content.count("\n") + 1 if content else 0
        )
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể đọc file {file_path.name}: {e}", file=sys.stderr)
        return None


def load_all_md_files(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_workers: int = 16,
    limit: Optional[int] = None,
    show_progress: bool = True
) -> List[PhoneDocument]:
    """
    Tải toàn bộ các file .md từ thư mục data_dir bằng đa luồng (multithreading).

    Args:
        data_dir: Thư mục chứa các file .md (mặc định: P:\\database phone cleaned)
        max_workers: Số luồng xử lý đồng thời (mặc định: 16)
        limit: Giới hạn số lượng file cần load (None để load toàn bộ)
        show_progress: Hiển thị thanh tiến trình tqdm (nếu có thư viện)

    Returns:
        Danh sách các PhoneDocument đã được load.
    """
    files = get_md_file_paths(data_dir)
    if limit is not None and limit > 0:
        files = files[:limit]

    total_files = len(files)
    documents: List[PhoneDocument] = []

    if total_files == 0:
        return documents

    use_tqdm = show_progress and (tqdm is not None)
    pbar = tqdm(total=total_files, desc="Đang tải .md", unit="file") if use_tqdm else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_single_md_file, f): f for f in files}
        for future in as_completed(futures):
            doc = future.result()
            if doc is not None:
                documents.append(doc)
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # Sắp xếp lại danh sách theo tên file để đảm bảo thứ tự nhất quán
    documents.sort(key=lambda d: d.filename)
    return documents


def stream_md_files(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    limit: Optional[int] = None
) -> Iterator[PhoneDocument]:
    """
    Generator đọc tuần tự từng file .md (tiết kiệm RAM nếu cần stream).
    """
    files = get_md_file_paths(data_dir)
    if limit is not None and limit > 0:
        files = files[:limit]

    for f in files:
        doc = read_single_md_file(f)
        if doc is not None:
            yield doc


def parse_phone_metadata(content: str) -> Dict[str, str]:
    """
    Hàm tiện ích: Trích xuất các trường thông tin cơ bản từ nội dung file .md.
    Ví dụ: Brand, Model, Released, Platform, CPU, RAM,...
    """
    metadata: Dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("[") or line.startswith("-"):
            continue
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip()
            if key and val:
                metadata[key] = val
    return metadata


def save_to_jsonl(documents: List[PhoneDocument], output_path: str | Path) -> None:
    """
    Lưu danh sách tài liệu sang định dạng JSON Lines (.jsonl).
    """
    import json
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(documents)} bản ghi vào: {out_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="B1: Load toàn bộ file .md từ database phone cleaned")
    parser.add_argument("--dir", type=str, default=DEFAULT_DATA_DIR, help="Đường dẫn thư mục chứa file .md")
    parser.add_argument("--workers", type=int, default=16, help="Số luồng đọc song song (mặc định: 16)")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số file cần đọc để test nhanh (mặc định: toàn bộ)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file .jsonl nếu muốn xuất dữ liệu")
    parser.add_argument("--no-progress", action="store_true", help="Tắt hiển thị thanh tiến trình")

    args = parser.parse_args()

    print("=" * 65)
    print(" BƯỚC 1: LOAD DỮ LIỆU CÁC FILE MARKDOWN (.MD)")
    print("=" * 65)
    print(f"Thư mục nguồn      : {args.dir}")
    print(f"Số luồng xử lý     : {args.workers}")
    if args.limit:
        print(f"Giới hạn load      : {args.limit} file")
    print("-" * 65)

    start_time = time.time()
    docs = load_all_md_files(
        data_dir=args.dir,
        max_workers=args.workers,
        limit=args.limit,
        show_progress=not args.no_progress
    )
    elapsed = time.time() - start_time

    total_bytes = sum(d.size_bytes for d in docs)
    total_chars = sum(d.char_count for d in docs)
    total_lines = sum(d.line_count for d in docs)
    total_mb = total_bytes / (1024 * 1024)
    speed_files = len(docs) / elapsed if elapsed > 0 else 0
    speed_mb = total_mb / elapsed if elapsed > 0 else 0

    print("-" * 65)
    print(" KẾT QUẢ TẢI DỮ LIỆU:")
    print(f" - Tổng số file đã load     : {len(docs):,} file")
    print(f" - Tổng dung lượng          : {total_mb:.2f} MB ({total_bytes:,} bytes)")
    print(f" - Tổng số ký tự            : {total_chars:,} ký tự")
    print(f" - Tổng số dòng             : {total_lines:,} dòng")
    print(f" - Thời gian thực thi       : {elapsed:.2f} giây")
    print(f" - Tốc độ đọc               : {speed_files:.1f} file/s (~{speed_mb:.2f} MB/s)")
    print("=" * 65)

    if docs:
        first_doc = docs[0]
        meta = parse_phone_metadata(first_doc.content)
        print(f"\n[XEM THỬ TÀI LIỆU ĐẦU TIÊN: {first_doc.filename}]")
        print(f"- Dung lượng: {first_doc.size_bytes} bytes | Dòng: {first_doc.line_count} | Ký tự: {first_doc.char_count}")
        print("- Metadata trích xuất nhanh:")
        for k in ["Brand", "Model", "Released", "Device Category", "Operating System", "RAM Capacity (converted)"]:
            if k in meta:
                print(f"  + {k}: {meta[k]}")
        print("\n- Trích đoạn nội dung (15 dòng đầu):")
        preview_lines = first_doc.content.strip().splitlines()[:15]
        for line in preview_lines:
            print(f"  | {line}")

    if args.output:
        save_to_jsonl(docs, args.output)

    print("\nHoàn tất B1.py.")


if __name__ == "__main__":
    main()
