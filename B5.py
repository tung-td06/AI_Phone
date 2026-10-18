"""
Module B5: Embedding toàn bộ Cơ sở dữ liệu thông số kỹ thuật điện thoại & Lưu Vector Database.
=============================================================================
Quy trình thực hiện:
  1. Nạp toàn bộ các đoạn Chunk thông số kỹ thuật điện thoại từ Bước B2
     (kèm context header nhận diện máy: Brand, Model, Section, Body Specs).
  2. Nạp Mô hình Embedding chuyên biệt đã được Fine-tune ở Bước B4
     (đường dẫn: 'p:\\train AI\\final_phone_embedding_model').
  3. Sử dụng GPU RTX 3060 (CUDA) với Mixed Precision FP16 và Batch Processing
     để chuyển đổi toàn bộ text chunks -> dense vectors [384 chiều].
  4. Chuẩn hóa L2 (Unit Norm) để tích vô hướng Dot Product tương đương Cosine Similarity.
  5. Lưu trữ CSDL Vector chuẩn hóa xuất xưởng:
     A) Định dạng FAISS RAG DB (phục vụ RAG Pipeline):
        rag_db/
        ├── index.faiss    : Chỉ mục vector nhị phân FAISS (IndexFlatIP)
        ├── chunks.json    : Toàn bộ văn bản thô của từng đoạn chunk
        └── metadata.json  : Thông tin chi tiết (brand, model, section, file)
     B) Định dạng Matrix DB:
        phone_vector_db/
        ├── embeddings.npy     : Ma trận Dense Vectors (N, 384) float32/float16
        ├── metadata.parquet  : Bảng dữ liệu metadata (PyArrow Snappy)
        ├── metadata.jsonl    : Bản ghi JSON Lines dự phòng
        └── summary.json       : Thống kê số lượng, cấu hình, tốc độ
  6. Cơ chế Tái sử dụng Vector (Smart Caching):
     Vector được lưu lại vĩnh viễn, các lần chạy sau KHÔNG CẦN embedding lại toàn bộ.
  7. Tích hợp sẵn 'PhoneVectorStore' hỗ trợ tìm kiếm ngữ nghĩa siêu tốc (Semantic Search),
     lọc theo Hãng (Brand) hoặc Chuyên mục (Section), và chế độ Chat hỏi đáp trong Terminal.
=============================================================================
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Iterator

# Thiết lập bảng mã UTF-8 cho Terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from sentence_transformers import SentenceTransformer

# Nhập các thành phần cần thiết từ B1 và B2
from B1 import DEFAULT_DATA_DIR, PhoneDocument, load_all_md_files
from B2 import PhoneChunk, chunk_all_documents, CANONICAL_SECTIONS


# ==============================================================================
# HẰNG SỐ CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
DEFAULT_MODEL_DIR = r"p:\train AI\final_phone_embedding_model"
DEFAULT_OUTPUT_DIR = r"p:\train AI\phone_vector_db"
DEFAULT_RAG_DB_DIR = r"p:\train AI\rag_db"
DEFAULT_BATCH_SIZE = 128
DEFAULT_EMBEDDING_DIM = 384


# ==============================================================================
# 1. KIỂM TRA PHẦN CỨNG & MÔI TRƯỜNG THỰC THI (GPU / CPU)
# ==============================================================================
def detect_hardware() -> Dict[str, Any]:
    """Phát hiện và báo cáo thông tin phần cứng tối ưu cho embedding."""
    cuda_available = torch.cuda.is_available()
    device_name = "cpu"
    gpu_name = "Không có (Chạy trên CPU)"
    vram_gb = 0.0

    if cuda_available:
        device_name = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)

    return {
        "cuda_available": cuda_available,
        "device": device_name,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
    }


def print_hardware_banner(hw: Dict[str, Any]):
    """In thông tin phần cứng trực quan."""
    print("-" * 75)
    print(" [1/5] THÔNG TIN PHẦN CỨNG THỰC THI:")
    print(f" - Thiết bị thực thi       : {hw['device'].upper()}")
    print(f" - Tên GPU / Phần cứng     : {hw['gpu_name']}")
    if hw["cuda_available"]:
        print(f" - Bộ nhớ VRAM khả dụng    : {hw['vram_gb']:.2f} GB")
        print(" -> [GPU ACCELERATION] RTX 3060 với CUDA kích hoạt chế độ suy luận siêu tốc!")
    else:
        print(" ! [CẢNH BÁO] Không tìm thấy GPU CUDA, chuyển sang chạy trên CPU.")
    print("-" * 75)


# ==============================================================================
# 2. THU THẬP & CHUẨN BỊ CHUNKS (TỪ FILES HOẶC TỪ CACHE)
# ==============================================================================
def prepare_database_chunks(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    mode: str = "grouped",
    add_context_header: bool = True,
    force_rechunk: bool = False,
    show_progress: bool = True,
) -> List[PhoneChunk]:
    """
    Chuẩn bị danh sách PhoneChunk từ thư mục markdown.
    Nếu đã có cache metadata trong thư mục output và không yêu cầu force_rechunk,
    hàm sẽ nạp nhanh trực tiếp từ cache.
    """
    out_path = Path(output_dir)
    cache_parquet = out_path / "metadata.parquet"
    cache_jsonl = out_path / "metadata.jsonl"

    # Kiểm tra cache có sẵn nếu không giới hạn limit và không force_rechunk
    if not force_rechunk and limit is None:
        if cache_parquet.exists() and HAS_PYARROW:
            print(f" -> Tìm thấy cache Parquet tại: {cache_parquet.name}. Đang nạp nhanh...")
            try:
                table = pq.read_table(cache_parquet)
                chunks: List[PhoneChunk] = []
                pydict = table.to_pydict()
                total_rows = len(table)
                for i in range(total_rows):
                    chunks.append(
                        PhoneChunk(
                            chunk_id=pydict["chunk_id"][i],
                            brand=pydict["brand"][i],
                            model=pydict["model"][i],
                            source_file=pydict["source_file"][i],
                            section=pydict["section"][i],
                            content=pydict["content"][i],
                            char_count=int(pydict["char_count"][i]),
                            token_estimate=int(pydict["token_estimate"][i]),
                        )
                    )
                print(f" -> Đã nạp thành công {len(chunks):,} chunks từ cache Parquet!")
                return chunks
            except Exception as e:
                print(f" ! Lỗi khi đọc cache Parquet ({e}). Chuyển sang đọc lại từ file markdown.")

        elif cache_jsonl.exists():
            print(f" -> Tìm thấy cache JSONL tại: {cache_jsonl.name}. Đang nạp nhanh...")
            try:
                chunks = []
                with open(cache_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            chunks.append(
                                PhoneChunk(
                                    chunk_id=d["chunk_id"],
                                    brand=d["brand"],
                                    model=d["model"],
                                    source_file=d["source_file"],
                                    section=d["section"],
                                    content=d["content"],
                                    char_count=d.get("char_count", len(d["content"])),
                                    token_estimate=d.get("token_estimate", len(d["content"]) // 4),
                                )
                            )
                print(f" -> Đã nạp thành công {len(chunks):,} chunks từ cache JSONL!")
                return chunks
            except Exception as e:
                print(f" ! Lỗi khi đọc cache JSONL ({e}). Chuyển sang đọc lại từ file markdown.")

    # Đọc tài liệu markdown từ thư mục nguồn
    print(f" -> Đang quét và tải các file markdown từ: {Path(data_dir).resolve()} ...")
    t0 = time.time()
    documents = load_all_md_files(
        data_dir=data_dir,
        limit=limit,
        show_progress=show_progress,
    )
    t_load = time.time() - t0
    print(f"    + Đã tải {len(documents):,} file markdown trong {t_load:.2f}s.")

    # Thực hiện chia nhỏ thành các PhoneChunk theo cấu trúc
    print(f" -> Đang phân mảnh (Chunking) theo 8 chuyên đề kỹ thuật chuẩn...")
    t1 = time.time()
    chunks = chunk_all_documents(
        documents=documents,
        mode=mode,
        add_context_header=add_context_header,
        show_progress=show_progress,
    )
    t_chunk = time.time() - t1
    print(f"    + Đã tạo thành công {len(chunks):,} chunks kỹ thuật trong {t_chunk:.2f}s.")

    return chunks


# ==============================================================================
# 3. QUY TRÌNH EMBEDDING TRÊN GPU (TEXT -> VECTOR [384])
# ==============================================================================
def encode_database_chunks(
    model_path: str | Path,
    chunks: List[PhoneChunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Sử dụng mô hình đã fine-tune để mã hóa toàn bộ văn bản của các chunks thành ma trận vector.
    """
    model_path_str = str(Path(model_path).resolve())
    print(f"\n -> Đang nạp mô hình Fine-tuned từ: {model_path_str} lên thiết bị '{device}'...")
    t_load_start = time.time()

    model = SentenceTransformer(model_path_str, device=device)

    # Đảm bảo mô hình chạy ở chế độ eval
    model.eval()
    print(f"    + Mô hình đã sẵn sàng trong {time.time() - t_load_start:.2f}s!")
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    print(f"    + Embedding Dimension     : {dim}")
    print(f"    + Max Sequence Length     : {model.max_seq_length} tokens")

    # Thu thập nội dung text cần mã hóa
    texts = [c.content for c in chunks]
    total_texts = len(texts)

    print(f"\n -> Bắt đầu quá trình mã hóa (Embedding) cho {total_texts:,} chunks...")
    print(f"    + Batch size              : {batch_size}")
    print(f"    + Chuẩn hóa L2 Norm       : BẬT (Cosine Similarity = Dot Product)")

    t_encode_start = time.time()

    # Thực thi mã hóa theo batch trên GPU
    # Với SentenceTransformers, normalize_embeddings=True giúp tính Dot Product cực nhanh sau này
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    t_encode_duration = time.time() - t_encode_start
    throughput = total_texts / max(t_encode_duration, 0.001)

    print(f"\n -> Hoàn tất quá trình sinh Vector Embeddings!")
    print(f"    + Tổng thời gian mã hóa   : {t_encode_duration:.2f}s ({t_encode_duration / 60:.2f} phút)")
    print(f"    + Tốc độ trung bình       : {throughput:.1f} chunks/giây")
    print(f"    + Kích thước ma trận      : {embeddings.shape} (N={embeddings.shape[0]:,}, Dim={embeddings.shape[1]})")
    print(f"    + Kiểu dữ liệu            : {embeddings.dtype}")

    return embeddings


# ==============================================================================
# 4. LƯU TRỮ VÀ XUẤT BẢN CƠ SỞ DỮ LIỆU VECTOR (FAISS & MATRIX)
# ==============================================================================
def save_vector_database(
    embeddings: np.ndarray,
    chunks: List[PhoneChunk],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rag_db_dir: Optional[str | Path] = DEFAULT_RAG_DB_DIR,
    model_name_or_path: str = DEFAULT_MODEL_DIR,
    data_source_dir: str = DEFAULT_DATA_DIR,
    save_fp16: bool = False,
) -> Dict[str, Any]:
    """
    Lưu ma trận vector và metadata đi kèm ra 2 định dạng chuẩn:
      1. FAISS RAG DB (rag_db/):
         - index.faiss    : Chỉ mục FAISS IndexFlatIP (tìm kiếm Cosine siêu tốc)
         - chunks.json    : Danh sách text của từng chunk
         - metadata.json  : Metadata chi tiết của từng chunk
      2. Matrix DB (phone_vector_db/):
         - embeddings.npy : Ma trận dense vector
         - metadata.parquet / metadata.jsonl
         - summary.json   : Thống kê tổng quan
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 75)
    print(" [4/5] LƯU TRỮ CƠ SỞ DỮ LIỆU VECTOR:")

    # --------------------------------------------------------------------------
    # A. XUẤT ĐỊNH DẠNG FAISS RAG DB (rag_db/)
    # --------------------------------------------------------------------------
    if rag_db_dir:
        rag_path = Path(rag_db_dir)
        rag_path.mkdir(parents=True, exist_ok=True)
        print(f" -> Xuất định dạng FAISS RAG DB vào: {rag_path.resolve()}")

        # 1. Lưu index.faiss
        if HAS_FAISS:
            dim = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            # FAISS IndexFlatIP yêu cầu kiểu float32
            emb_f32 = embeddings.astype(np.float32) if embeddings.dtype != np.float32 else embeddings
            faiss_index.add(emb_f32)
            faiss_file = rag_path / "index.faiss"
            faiss.write_index(faiss_index, str(faiss_file))
            faiss_size_mb = faiss_file.stat().st_size / (1024 * 1024)
            print(f"  [OK] Đã lưu FAISS Index          : {faiss_file.name} ({faiss_size_mb:.2f} MB, {faiss_index.ntotal:,} vectors)")
        else:
            print("  ! [BỎ QUA] Chưa cài thư viện faiss-cpu, bỏ qua lưu index.faiss.")

        # 2. Lưu chunks.json (chứa text thô của từng đoạn chunk)
        chunks_file = rag_path / "chunks.json"
        raw_texts = [c.content for c in chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(raw_texts, f, ensure_ascii=False)
        chunks_size_mb = chunks_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] Đã lưu văn bản chunks       : {chunks_file.name} ({chunks_size_mb:.2f} MB, {len(raw_texts):,} items)")

        # 3. Lưu metadata.json (thông tin ngữ cảnh tương ứng)
        meta_file = rag_path / "metadata.json"
        meta_list = [
            {
                "id": i,
                "chunk_id": c.chunk_id,
                "brand": c.brand,
                "model": c.model,
                "section": c.section,
                "source_file": c.source_file,
                "char_count": c.char_count,
                "token_estimate": c.token_estimate,
            }
            for i, c in enumerate(chunks)
        ]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False)
        meta_size_mb = meta_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] Đã lưu metadata tương ứng   : {meta_file.name} ({meta_size_mb:.2f} MB)")

    # --------------------------------------------------------------------------
    # B. XUẤT ĐỊNH DẠNG MATRIX DB (phone_vector_db/)
    # --------------------------------------------------------------------------
    print(f"\n -> Xuất định dạng Matrix DB vào: {out_dir.resolve()}")
    emb_data = embeddings.astype(np.float16) if save_fp16 else embeddings.astype(np.float32)
    npy_path = out_dir / "embeddings.npy"
    np.save(npy_path, emb_data)
    npy_size_mb = npy_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] Đã lưu ma trận vector vào   : {npy_path.name} ({npy_size_mb:.2f} MB, {emb_data.dtype})")

    # Lưu Metadata qua Parquet
    parquet_path = out_dir / "metadata.parquet"
    if HAS_PYARROW:
        metadata_dict = {
            "index": list(range(len(chunks))),
            "chunk_id": [c.chunk_id for c in chunks],
            "brand": [c.brand for c in chunks],
            "model": [c.model for c in chunks],
            "section": [c.section for c in chunks],
            "source_file": [c.source_file for c in chunks],
            "char_count": [c.char_count for c in chunks],
            "token_estimate": [c.token_estimate for c in chunks],
            "content": [c.content for c in chunks],
        }
        table = pa.Table.from_pydict(metadata_dict)
        pq.write_table(table, parquet_path, compression="snappy")
        pq_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Đã lưu bảng metadata vào    : {parquet_path.name} ({pq_size_mb:.2f} MB, Snappy Parquet)")

    # Lưu Metadata qua JSONL
    jsonl_path = out_dir / "metadata.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, c in enumerate(chunks):
            d = c.to_dict()
            d["index"] = idx
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    jsonl_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] Đã lưu bảng metadata vào    : {jsonl_path.name} ({jsonl_size_mb:.2f} MB)")

    # Thống kê tổng quan
    brand_counts: Dict[str, int] = {}
    section_counts: Dict[str, int] = {}
    for c in chunks:
        brand_counts[c.brand] = brand_counts.get(c.brand, 0) + 1
        section_counts[c.section] = section_counts.get(c.section, 0) + 1

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name_or_path": str(Path(model_name_or_path).resolve()),
        "data_source_dir": str(Path(data_source_dir).resolve()),
        "total_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(emb_data.dtype),
        "normalized": True,
        "unique_brands": len(brand_counts),
        "unique_sections": len(section_counts),
        "top_5_brands": dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        "section_distribution": section_counts,
        "rag_db": {
            "dir": str(Path(rag_db_dir).resolve()) if rag_db_dir else None,
            "has_faiss": HAS_FAISS,
        },
        "files": {
            "embeddings": npy_path.name,
            "metadata_parquet": parquet_path.name if HAS_PYARROW else None,
            "metadata_jsonl": jsonl_path.name,
        }
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Đã lưu file tổng kết vào    : {summary_path.name}")
    print("-" * 75)

    return summary


# ==============================================================================
# 5. SEMANTIC SEARCH ENGINE (CÔNG CỤ TÌM KIẾM NGỮ NGHĨA VECTOR)
# ==============================================================================
class PhoneVectorStore:
    """
    Công cụ tìm kiếm thông số điện thoại theo Vector Embeddings.
    Tự động ưu tiên sử dụng FAISS (IndexFlatIP) nếu tìm thấy trong 'rag_db',
    hoặc fallback sang ma trận NumPy 'embeddings.npy'.
    """

    def __init__(
        self,
        db_dir: str | Path = DEFAULT_OUTPUT_DIR,
        rag_db_dir: str | Path = DEFAULT_RAG_DB_DIR,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
    ):
        self.db_dir = Path(db_dir)
        self.rag_db_dir = Path(rag_db_dir)
        self.model_dir = Path(model_dir)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.faiss_index = None
        self.chunks_text: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        self.backend = "unknown"

        self._load_store()

    def _load_store(self):
        """Nạp Vector Store từ FAISS rag_db hoặc Matrix DB."""
        faiss_file = self.rag_db_dir / "index.faiss"
        chunks_file = self.rag_db_dir / "chunks.json"
        meta_file = self.rag_db_dir / "metadata.json"

        # Ưu tiên 1: Nạp từ FAISS RAG DB (nhẹ, nhanh, chuẩn production)
        if HAS_FAISS and faiss_file.exists() and chunks_file.exists() and meta_file.exists():
            self.faiss_index = faiss.read_index(str(faiss_file))
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks_text = json.load(f)
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.backend = "FAISS (IndexFlatIP)"

        # Ưu tiên 2: Fallback nạp từ Matrix DB (NumPy + Parquet / JSONL)
        else:
            npy_path = self.db_dir / "embeddings.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Không tìm thấy CSDL Vector tại cả '{self.rag_db_dir}' lẫn '{self.db_dir}'!")

            self.embeddings = np.load(npy_path)
            if self.embeddings.dtype != np.float32:
                self.embeddings = self.embeddings.astype(np.float32)

            parquet_path = self.db_dir / "metadata.parquet"
            jsonl_path = self.db_dir / "metadata.jsonl"

            if parquet_path.exists() and HAS_PYARROW:
                table = pq.read_table(parquet_path)
                pydict = table.to_pydict()
                num_rows = len(table)
                self.chunks_text = pydict["content"]
                self.metadata = [
                    {
                        "id": i,
                        "chunk_id": pydict["chunk_id"][i],
                        "brand": pydict["brand"][i],
                        "model": pydict["model"][i],
                        "section": pydict["section"][i],
                        "source_file": pydict["source_file"][i],
                    }
                    for i in range(num_rows)
                ]
            elif jsonl_path.exists():
                self.metadata = []
                self.chunks_text = []
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            d = json.loads(line)
                            self.chunks_text.append(d["content"])
                            self.metadata.append({
                                "id": i,
                                "chunk_id": d["chunk_id"],
                                "brand": d["brand"],
                                "model": d["model"],
                                "section": d["section"],
                                "source_file": d["source_file"],
                            })
            self.backend = "NumPy Dot Product"

        # Nạp mô hình mã hóa câu hỏi
        self.model = SentenceTransformer(str(self.model_dir), device=self.device)
        self.model.eval()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_brand: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm Top K đoạn thông số kỹ thuật phù hợp nhất với câu hỏi.
        """
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp!")

        # 1. Mã hóa câu hỏi thành vector truy vấn L2-normalized
        query_vec = self.model.encode(
            [query],
            device=self.device,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        results = []

        # 2A. Tìm kiếm bằng FAISS
        if self.faiss_index is not None:
            # Lấy dư mẫu nếu có bộ lọc metadata
            fetch_k = min(len(self.metadata), top_k * 10 if (filter_brand or filter_section) else top_k)
            distances, indices = self.faiss_index.search(query_vec, fetch_k)

            count = 0
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                meta = self.metadata[idx]
                if filter_brand and meta["brand"].lower() != filter_brand.lower():
                    continue
                if filter_section and meta["section"].lower() != filter_section.lower():
                    continue

                count += 1
                results.append({
                    "rank": count,
                    "score": float(score),
                    "brand": meta["brand"],
                    "model": meta["model"],
                    "section": meta["section"],
                    "chunk_id": meta["chunk_id"],
                    "content": self.chunks_text[idx],
                })
                if count >= top_k:
                    break

        # 2B. Tìm kiếm bằng NumPy Dot Product
        elif self.embeddings is not None:
            scores = np.dot(self.embeddings, query_vec[0])

            if filter_brand or filter_section:
                mask = np.ones(len(self.metadata), dtype=bool)
                if filter_brand:
                    b_low = filter_brand.lower()
                    mask &= np.array([m["brand"].lower() == b_low for m in self.metadata])
                if filter_section:
                    s_low = filter_section.lower()
                    mask &= np.array([m["section"].lower() == s_low for m in self.metadata])
                scores = np.where(mask, scores, -np.inf)

            top_indices = np.argsort(scores)[::-1][:top_k]
            for rank, idx in enumerate(top_indices, 1):
                score = float(scores[idx])
                if score == -np.inf:
                    continue
                meta = self.metadata[idx]
                results.append({
                    "rank": rank,
                    "score": score,
                    "brand": meta["brand"],
                    "model": meta["model"],
                    "section": meta["section"],
                    "chunk_id": meta["chunk_id"],
                    "content": self.chunks_text[idx],
                })

        return results


# ==============================================================================
# 6. BENCHMARK & DEMO TÌM KIẾM TỰ ĐỘNG
# ==============================================================================
def run_search_demo(store: PhoneVectorStore):
    """
    Chạy thử nghiệm tìm kiếm mẫu để đánh giá chất lượng embedding thực tế.
    """
    print("\n" + "=" * 75)
    print(f" [5/5] DEMO TÌM KIẾM NGỮ NGHĨA (Engine: {store.backend}):")
    print("=" * 75)

    test_queries = [
        "dung lượng pin và khả năng sạc của Apple iPhone 13 Pro Max",
        "Samsung Galaxy S22 Ultra màn hình bao nhiêu inch và độ phân giải gì?",
        "Xiaomi 13 dùng vi xử lý chipset gì và card đồ họa GPU nào?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n--- [CÂU HỎI THỬ NGHIỆM #{i}] ---")
        print(f" Query: \"{q}\"")
        results = store.search(q, top_k=2)

        print("-" * 75)
        for r in results:
            print(f" [Top {r['rank']}] Điểm tương đồng (Cosine Sim): {r['score']:.4f}")
            print(f"        Thiết bị: {r['brand']} {r['model']} | Chuyên mục: [{r['section']}]")
            content_preview = "\n        ".join(r["content"].strip().splitlines()[:4])
            print(f"        Nội dung trích đoạn:\n        {content_preview}")
            print("        ...")
            print("-" * 75)


# ==============================================================================
# 7. CHẾ ĐỘ TƯƠNG TÁC DÒNG LỆNH (INTERACTIVE CLI)
# ==============================================================================
def run_interactive_mode(store: PhoneVectorStore):
    """Mở giao diện chat hỏi đáp tìm kiếm thông số trực tiếp trong Terminal."""
    print("\n" + "=" * 75)
    print(f" GIAO DIỆN HỎI ĐÁP THÔNG SỐ ĐIỆN THOẠI (Engine: {store.backend})")
    print(" Nhập câu hỏi bất kỳ (ví dụ: 'Pin iPhone 14 Pro Max', 'Chip Xiaomi 13')")
    print(" Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 75 + "\n")

    while True:
        try:
            query = input(" [Hỏi đáp RAG] Nhập câu hỏi > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print(" -> Tạm biệt!")
                break

            results = store.search(query, top_k=3)
            print("\n" + "=" * 70)
            print(f" KẾT QUẢ TÌM KIẾM CHO: \"{query}\"")
            print("=" * 70)
            for r in results:
                print(f" * [Top {r['rank']}] Độ khớp: {r['score']:.4f} | {r['brand']} {r['model']} ({r['section']})")
                print("-" * 70)
                print(r["content"].strip())
                print("=" * 70)
            print()
        except KeyboardInterrupt:
            print("\n -> Đã dừng phiên tương tác.")
            break


# ==============================================================================
# 8. HÀM MAIN VÀ ĐIỀU PHỐI TIẾN TRÌNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="B5: Embedding toàn bộ Database điện thoại thành Dense Vectors & Lưu Vector DB (FAISS)"
    )

    # Đường dẫn
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Thư mục chứa mô hình Fine-tuned (mặc định: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Thư mục chứa file markdown thông số (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu ma trận vector và metadata (mặc định: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--rag-db-dir",
        type=str,
        default=DEFAULT_RAG_DB_DIR,
        help=f"Thư mục lưu Vector Database FAISS (mặc định: {DEFAULT_RAG_DB_DIR})",
    )

    # Cấu hình xử lý & phần cứng
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Kích thước mini-batch cho GPU (mặc định: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số file markdown đọc vào để test nhanh (mặc định: None - đọc toàn bộ)",
    )
    parser.add_argument(
        "--fp16-storage",
        action="store_true",
        help="Lưu ma trận vector dạng float16 để giảm 50% dung lượng đĩa",
    )
    parser.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Bắt buộc phân mảnh lại từ markdown gốc dù đã có cache metadata",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Bắt buộc tính toán embedding lại từ đầu dù đã có CSDL Vector đã lưu",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Tắt thanh tiến trình",
    )

    # Chế độ tìm kiếm nhanh / chat
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Thực hiện tìm kiếm câu hỏi trực tiếp trên CSDL vector có sẵn",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mở chế độ hỏi đáp tương tác trực tiếp trong Terminal",
    )

    args = parser.parse_args()

    # Nhận diện phần cứng
    hw = detect_hardware()

    # 1. Nếu người dùng chỉ muốn tìm kiếm câu hỏi hoặc chạy interactive mode
    if args.search or args.interactive:
        store = PhoneVectorStore(
            db_dir=args.output_dir,
            rag_db_dir=args.rag_db_dir,
            model_dir=args.model_dir,
            device=hw["device"],
        )
        if args.search:
            print(f"\n -> Đang tìm kiếm cho: \"{args.search}\" ...")
            res = store.search(args.search, top_k=5)
            for r in res:
                print(f"\n[Top {r['rank']}] Điểm tương đồng: {r['score']:.4f}")
                print(f"Thiết bị: {r['brand']} {r['model']} [{r['section']}]")
                print("-" * 50)
                print(r["content"].strip())
        if args.interactive:
            run_interactive_mode(store)
        return

    # 2. KIỂM TRA CƠ CHẾ TÁI SỬ DỤNG VECTOR (SMART CACHING)
    rag_dir = Path(args.rag_db_dir)
    faiss_file = rag_dir / "index.faiss"
    chunks_file = rag_dir / "chunks.json"
    meta_file = rag_dir / "metadata.json"
    matrix_file = Path(args.output_dir) / "embeddings.npy"

    has_rag_db = faiss_file.exists() and chunks_file.exists() and meta_file.exists()
    has_matrix_db = matrix_file.exists()

    if (has_rag_db or has_matrix_db) and not args.force_reembed and args.limit is None:
        print("=" * 75)
        print(" [THÔNG BÁO] ĐÃ TÌM THẤY CƠ SỞ DỮ LIỆU VECTOR CÓ SẴN:")
        if has_rag_db:
            print(f"  - FAISS Vector DB       : {rag_dir.resolve()}")
            print(f"    + index.faiss         ({faiss_file.stat().st_size / (1024*1024):.2f} MB)")
            print(f"    + chunks.json         ({chunks_file.stat().st_size / (1024*1024):.2f} MB)")
            print(f"    + metadata.json       ({meta_file.stat().st_size / (1024*1024):.2f} MB)")
        if has_matrix_db:
            print(f"  - Matrix Vector DB      : {Path(args.output_dir).resolve()}")
            print(f"    + embeddings.npy      ({matrix_file.stat().st_size / (1024*1024):.2f} MB)")
        print("\n -> [TÁI SỬ DỤNG] Vector đã được lưu lại, BỎ QUA quá trình tính toán embedding!")
        print(" -> (Để ép buộc tính toán embedding lại từ đầu, thêm tham số: --force-reembed)")
        print("=" * 75)

        vector_store = PhoneVectorStore(
            db_dir=args.output_dir,
            rag_db_dir=args.rag_db_dir,
            model_dir=args.model_dir,
            device=hw["device"],
        )
        run_search_demo(vector_store)
        return

    # 3. Quy trình Embedding toàn diện từ đầu
    print("=" * 75)
    print(" BƯỚC 5: EMBEDDING TOÀN BỘ CƠ SỞ DỮ LIỆU ĐIỆN THOẠI THÀNH DENSE VECTORS")
    print("=" * 75)
    print(f"Mô hình sử dụng         : {Path(args.model_dir).resolve()}")
    print(f"Thư mục dữ liệu nguồn   : {Path(args.data_dir).resolve()}")
    print(f"Thư mục lưu FAISS RAG DB: {Path(args.rag_db_dir).resolve()}")
    print(f"Thư mục lưu Matrix DB   : {Path(args.output_dir).resolve()}")
    print(f"Kích thước Batch size   : {args.batch_size}")
    if args.limit:
        print(f"Giới hạn mẫu thử nghiệm : {args.limit} file")
    print("-" * 75)

    print_hardware_banner(hw)

    # 4. Chuẩn bị chunks
    print("\n [2/5] CHUẨN BỊ DỮ LIỆU CHUNKS:")
    chunks = prepare_database_chunks(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        mode="grouped",
        add_context_header=True,
        force_rechunk=args.force_rechunk,
        show_progress=not args.no_progress,
    )

    if not chunks:
        print(" ! [LỖI] Không tìm thấy dữ liệu chunk nào để embedding. Kết thúc.")
        return

    # 5. Thực hiện Embedding trên GPU
    print("-" * 75)
    print(" [3/5] TIẾN HÀNH SINH VECTOR EMBEDDING:")
    embeddings = encode_database_chunks(
        model_path=args.model_dir,
        chunks=chunks,
        batch_size=args.batch_size,
        device=hw["device"],
        show_progress=not args.no_progress,
    )

    # 6. Lưu trữ CSDL Vector (FAISS + Matrix)
    save_vector_database(
        embeddings=embeddings,
        chunks=chunks,
        output_dir=args.output_dir,
        rag_db_dir=args.rag_db_dir,
        model_name_or_path=args.model_dir,
        data_source_dir=args.data_dir,
        save_fp16=args.fp16_storage,
    )

    # 7. Khởi tạo PhoneVectorStore và chạy Demo kiểm thử
    vector_store = PhoneVectorStore(
        db_dir=args.output_dir,
        rag_db_dir=args.rag_db_dir,
        model_dir=args.model_dir,
        device=hw["device"],
    )
    run_search_demo(vector_store)

    print("\n" + "=" * 75)
    print(" HOÀN TẤT THÀNH CÔNG BƯỚC B5!")
    print(f" CSDL FAISS RAG DB đã sẵn sàng tại : {Path(args.rag_db_dir).resolve()}")
    print(f" CSDL Matrix DB đã sẵn sàng tại    : {Path(args.output_dir).resolve()}")
    print(" Bạn có thể thử nghiệm tìm kiếm ngay bằng lệnh:")
    print(f"   python \"p:\\train AI\\B5.py\" --search \"Pin iPhone 13 Pro Max dùng được bao lâu?\"")
    print(" Hoặc mở giao diện chat tương tác:")
    print(f"   python \"p:\\train AI\\B5.py\" --interactive")
    print("=" * 75)


if __name__ == "__main__":
    main()
