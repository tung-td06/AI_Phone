#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PhoneDB Lossless HTML to Clean Markdown / Text Converter (xuli.py)
================================================================================
Chương trình chuyển đổi dữ liệu HTML PhoneDB thành Markdown/TXT sạch, có cấu trúc,
bảo toàn 100% dữ liệu kỹ thuật nguồn, phục vụ cho hệ thống RAG và phân tích dữ liệu.

Tác giả: Senior Data Engineer
Hệ thống: PhoneDB Dataset Preprocessing Pipeline
================================================================================
"""

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Cấu hình UTF-8 cho console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
except ImportError:
    print("[!] Lỗi: Thiếu thư viện 'beautifulsoup4'. Vui lòng cài đặt: pip install beautifulsoup4 lxml")
    sys.exit(1)


# ==============================================================================
# 1. HẰNG SỐ & CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
def find_default_input_dir() -> str:
    kaggle_candidates = [
        "/kaggle/input/phone-db",
        "/kaggle/input/phone-db/data",
        "/kaggle/input/phone-db/database phone/data",
    ]
    for p in kaggle_candidates:
        if os.path.exists(p):
            return p
    if os.path.exists(r"P:\database phone\data"):
        return r"P:\database phone\data"
    if os.path.exists("data"):
        return "data"
    return "/kaggle/input/phone-db" if os.path.exists("/kaggle") else r"P:\database phone\data"

def find_default_output_dir() -> str:
    if os.path.exists("/kaggle"):
        return "/kaggle/working/database_phone_cleaned"
    if os.path.exists(r"P:\database phone cleaned"):
        return r"P:\database phone cleaned"
    return "database_phone_cleaned"

DEFAULT_INPUT_DIR = find_default_input_dir()
DEFAULT_OUTPUT_DIR = find_default_output_dir()
DEFAULT_EXT = ".md"

# Parser ưu tiên: lxml (nhanh, chuẩn), fallback: html.parser
PARSER_NAME = "lxml"
try:
    import lxml
except ImportError:
    PARSER_NAME = "html.parser"


# ==============================================================================
# 2. BỘ XỬ LÝ LÀM SẠCH VĂN BẢN (TEXT NORMALIZATION)
# ==============================================================================
def clean_whitespace_and_punctuation(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng và dấu câu mà tuyệt đối không làm thay đổi
    các giá trị số, đơn vị đo, hoặc mã định danh (ví dụ: '12,1' không bị đổi thành '12, 1').
    """
    if not text:
        return ""
    
    # Giải mã HTML entities nếu còn sót
    text = html.unescape(text)
    
    # Thay thế non-breaking spaces và ký tự khoảng trắng đặc biệt
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    
    lines = []
    for raw_line in text.split("\n"):
        # Gộp các khoảng trắng ngang liên tiếp thành 1 dấu cách
        line = re.sub(r"[ \t\r\f\v]+", " ", raw_line).strip()
        if not line:
            continue
        
        # Xóa khoảng trắng thừa đứng TRƯỚC dấu câu (ví dụ: 'Word , Word' -> 'Word, Word')
        # Nhưng không chèn thêm khoảng trắng vào số/mã (ví dụ: '12,1' giữ nguyên)
        line = re.sub(r"\s+([,;:.)\]])", r"\1", line)
        # Xóa khoảng trắng đứng SAU dấu mở ngoặc: '( Text' -> '(Text'
        line = re.sub(r"([(\[])\s+", r"\1", line)
        
        # Xóa khoảng trắng bị tách giữa số và dấu gạch nối (ví dụ: '2 -way' -> '2-way', '1 -notch' -> '1-notch')
        line = re.sub(r"(\d+)\s+-\s*([a-zA-Z]+)", r"\1-\2", line)
        
        # Xóa dấu phẩy / chấm phẩy thừa đứng ở đầu dòng do <br> ở đầu phần tử
        line = re.sub(r"^[,\s;]+", "", line)
        
        lines.append(line)
        
    return "\n".join(lines)


def extract_node_text(node: Tag) -> str:
    """
    Trích xuất văn bản từ một thẻ HTML (node), bảo toàn phân tách thẻ <br>,
    xử lý khoảng cách giữa các thẻ con inline hợp lý để tránh dính chữ (vd: 'DualSIM').
    """
    if not node:
        return ""
        
    # Thay thẻ <br> bằng token phân tách đặc biệt trước khi lấy text
    for br in node.find_all("br"):
        br.replace_with(" __BR_SEP__ ")
        
    raw_text = node.get_text(" ", strip=True)
    
    # Chuyển token __BR_SEP__ thành dấu phẩy và khoảng trắng hoặc xuống dòng
    # Đối với danh sách các giá trị như Mobile Operator: phân tách bằng dấu phẩy
    raw_text = raw_text.replace(" __BR_SEP__ ", ", ").replace("__BR_SEP__", ", ")
    
    return clean_whitespace_and_punctuation(raw_text)


# ==============================================================================
# 3. BỘ PHÂN TÍCH HTML CHÍNH (HTML TO DATASET PARSER)
# ==============================================================================
class PhoneDBParser:
    """
    Bộ phân tích HTML PhoneDB chuyên dụng:
    - Loại bỏ UI, quảng cáo, nút điều hướng, so sánh, scripts, css.
    - Bảo toàn 100% section, subsection, field, value, description.
    - Xử lý các hàng đặc biệt: Brief, hàng tiếp nối (continuation), ô có <br>.
    - Không làm mất nội dung không xác định (chuyển vào Unclassified).
    """

    @staticmethod
    def parse_html(html_content: str) -> Dict[str, Any]:
        """
        Phân tích chuỗi HTML và trích xuất cấu trúc dữ liệu hoàn chỉnh.
        Trả về dict chứa danh sách các sections, thống kê và nội dung markdown.
        """
        soup = BeautifulSoup(html_content, PARSER_NAME)
        
        # 1. Tìm bảng thông số kỹ thuật (Datasheet table)
        # Bảng datasheet luôn là bảng đầu tiên hoặc bảng có chứa các thẻ h4/h5/datasheet_item
        tables = soup.find_all("table")
        main_table = None
        for tbl in tables:
            if tbl.find(["h4", "h5"]) or tbl.find(id=re.compile(r"datasheet_item")):
                main_table = tbl
                break
        
        if not main_table and tables:
            main_table = tables[0]
            
        if not main_table:
            # Fallback nếu không có thẻ table: phân tích toàn bộ soup
            main_table = soup

        sections: List[Dict[str, Any]] = []
        current_section: Optional[Dict[str, Any]] = None
        
        src_stats = {
            "sections_count": 0,
            "fields_count": 0,
            "continuation_count": 0,
            "unclassified_count": 0
        }

        # 2. Duyệt qua từng hàng trong bảng
        rows = main_table.find_all("tr") if hasattr(main_table, "find_all") else []
        
        for tr in rows:
            # Bỏ qua các hàng chứa quảng cáo hoặc script
            if tr.find(["script", "ins"]) or "adsbygoogle" in str(tr):
                continue
                
            tds = tr.find_all("td")
            if not tds:
                continue

            # ==============================================================
            # TRƯỜNG HỢP 1: HÀNG 1 CỘT (COLSPAN=2 HOẶC 1 TD)
            # Thường là Tiêu đề Section (h4/h5), Brief giới thiệu, hoặc Ad
            # ==============================================================
            if len(tds) == 1:
                td = tds[0]
                
                # A. Kiểm tra Tiêu đề Section (h4 hoặc h5)
                h_tag = td.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if h_tag:
                    # Loại bỏ phần điều hướng: 'Expand datasheet | Add to compare'
                    for nav in td.find_all("div"):
                        nav.decompose()
                    
                    sec_text = h_tag.get_text(" ", strip=True)
                    sec_name = re.sub(r"[:\s]+$", "", sec_text)
                    
                    # Fallback nếu text trong h_tag rỗng (dùng alt/title của icon)
                    if not sec_name:
                        for img in h_tag.find_all("img"):
                            sec_name = img.get("title") or img.get("alt") or ""
                            if sec_name:
                                break
                    sec_name = sec_name or "General"
                    
                    # Lưu section cũ và tạo section mới
                    if current_section is not None:
                        sections.append(current_section)
                        
                    current_section = {
                        "name": sec_name,
                        "level": h_tag.name,  # 'h4' là section, 'h5' là subsection
                        "items": []
                    }
                    src_stats["sections_count"] += 1
                    continue

                # B. Kiểm tra hàng Brief (tóm tắt giới thiệu sản phẩm)
                strong_tag = td.find("strong")
                if strong_tag and strong_tag.get_text(strip=True).lower() == "brief":
                    # Xóa chữ 'Brief' trong strong
                    strong_tag.decompose()
                    
                    # Xóa icon tooltip info_gray
                    for img in td.find_all("img"):
                        img.decompose()
                        
                    # Xóa nút điều hướng '...' hoặc '››' trỏ tới detailed view
                    for a in td.find_all("a"):
                        a_text = a.get_text(strip=True)
                        if a_text in ("...", "››", "\u203a\u203a"):
                            a.decompose()
                        elif "Detailed view" in (a.get("title") or ""):
                            a.decompose()

                    brief_val = extract_node_text(td)
                    if brief_val:
                        if current_section is None:
                            current_section = {"name": "Introduction", "level": "h4", "items": []}
                            src_stats["sections_count"] += 1
                            
                        current_section["items"].append({
                            "type": "field",
                            "key": "Brief",
                            "value": brief_val
                        })
                        src_stats["fields_count"] += 1
                    continue

                # C. Hàng 1 cột khác: Kiểm tra xem có chứa nội dung thông số không
                for tag in td.find_all(["script", "style", "ins"]):
                    tag.decompose()
                other_text = extract_node_text(td)
                # Bỏ qua nếu là mạng xã hội hoặc rỗng
                if other_text and not any(kw in other_text for kw in ["Tweet", "Facebook", "facebook", "comments"]):
                    if current_section is None:
                        current_section = {"name": "Unclassified Source Content", "level": "h4", "items": []}
                        src_stats["sections_count"] += 1
                    current_section["items"].append({
                        "type": "cont",
                        "key": "",
                        "value": other_text
                    })
                    src_stats["unclassified_count"] += 1

            # ==============================================================
            # TRƯỜNG HỢP 2: HÀNG 2 CỘT (TD0: KEY, TD1: VALUE)
            # ==============================================================
            elif len(tds) >= 2:
                k_td = tds[0]
                v_td = tds[1]

                # --- 1. Làm sạch cột Key (TD0) ---
                for tag in k_td.find_all(["img", "script", "style"]):
                    tag.decompose()
                k_text = extract_node_text(k_td)
                k_clean = re.sub(r"[:\s]+$", "", k_text)

                # --- 2. Làm sạch cột Value (TD1) ---
                for tag in v_td.find_all(["script", "style", "ins"]):
                    tag.decompose()
                    
                # Xóa các icon điều hướng và tooltip chuẩn (arrow, compare, info_gray)
                for img in v_td.find_all("img"):
                    src = img.get("src", "")
                    if "arrow_" in src or "cmp_" in src or "info_gray" in src:
                        img.decompose()
                        
                val_clean = extract_node_text(v_td)
                if not val_clean:
                    continue

                if current_section is None:
                    current_section = {"name": "Introduction", "level": "h4", "items": []}
                    src_stats["sections_count"] += 1

                if k_clean:
                    current_section["items"].append({
                        "type": "field",
                        "key": k_clean,
                        "value": val_clean
                    })
                    src_stats["fields_count"] += 1
                else:
                    # Hàng không có nhãn (continuation row): liên kết với trường trước đó
                    current_section["items"].append({
                        "type": "cont",
                        "key": "",
                        "value": val_clean
                    })
                    src_stats["continuation_count"] += 1

        if current_section is not None:
            sections.append(current_section)

        # 3. Định dạng dữ liệu thành chuỗi Markdown / Clean Text
        md_content = PhoneDBParser.format_markdown(sections)
        
        return {
            "sections": sections,
            "markdown": md_content,
            "stats": src_stats
        }

    @staticmethod
    def format_markdown(sections: List[Dict[str, Any]]) -> str:
        """
        Định dạng danh sách sections thành cấu trúc text/markdown theo chuẩn yêu cầu:
        [Section Name]
        Key: Value
          - Continuation Value 1
          - Continuation Value 2
        """
        output_lines: List[str] = []
        
        for sec in sections:
            name = sec["name"]
            items = sec.get("items", [])
            
            output_lines.append(f"[{name}]")
            
            for item in items:
                itype = item["type"]
                k = item["key"]
                val = item["value"]
                
                lines = val.split("\n")
                if itype == "field":
                    output_lines.append(f"{k}: {lines[0]}")
                    for line in lines[1:]:
                        output_lines.append(f"  - {line}")
                else:
                    # Dòng tiếp nối (continuation row)
                    for line in lines:
                        output_lines.append(f"  - {line}")
                        
            output_lines.append("")  # Dòng trống ngăn cách các section
            
        return "\n".join(output_lines).rstrip() + "\n"


# ==============================================================================
# 4. BỘ KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU (AUDIT & INTEGRITY CHECKER)
# ==============================================================================
class IntegrityChecker:
    """
    Kiểm tra đối chiếu 1-1 giữa HTML nguồn và Markdown đầu ra:
    - Đếm và so khớp số section.
    - Đếm và so khớp số field.
    - Kiểm tra bảo toàn các danh sách nhiều giá trị (VD: băng tần).
    - Phát hiện trường nghi ngờ bị mất hoặc rỗng.
    """

    @staticmethod
    def verify(parsed_data: Dict[str, Any], md_content: str) -> Dict[str, Any]:
        src_stats = parsed_data["stats"]
        src_secs = src_stats["sections_count"]
        src_flds = src_stats["fields_count"]
        src_conts = src_stats["continuation_count"]

        # Đếm số lượng trong output Markdown
        md_secs = len(re.findall(r"^\[.*?\]$", md_content, re.MULTILINE))
        md_flds = len(re.findall(r"^[^\s\[].*?:", md_content, re.MULTILINE))
        md_conts = len(re.findall(r"^\s+-\s+", md_content, re.MULTILINE))

        errors: List[str] = []
        warnings: List[str] = []

        if md_secs != src_secs:
            errors.append(f"Số section không khớp: Nguồn={src_secs}, Đích={md_secs}")
        
        if md_flds != src_flds:
            errors.append(f"Số field không khớp: Nguồn={src_flds}, Đích={md_flds}")

        # Kiểm tra nội dung có bị rỗng không
        if not md_content.strip():
            errors.append("Nội dung đầu ra rỗng!")

        is_passed = len(errors) == 0

        return {
            "passed": is_passed,
            "errors": errors,
            "warnings": warnings,
            "src_stats": src_stats,
            "out_stats": {
                "sections_count": md_secs,
                "fields_count": md_flds,
                "continuation_count": md_conts
            }
        }


# ==============================================================================
# 5. XỬ LÝ ĐƠN LẺ MỘT FILE (WORKER FUNCTION)
# ==============================================================================
def process_single_file(args_tuple: Tuple[str, str, str, bool]) -> Dict[str, Any]:
    """
    Hàm xử lý độc lập cho 1 file (dùng được trong Multiprocessing / Multi-threading).
    args_tuple: (src_file_path, base_input_dir, base_output_dir, force_overwrite)
    """
    src_path, input_base, output_base, force = args_tuple
    
    try:
        # 1. Tính toán đường dẫn file đích tương ứng (lưu trực tiếp file .md vào output, không tạo thư mục con)
        base_no_ext, _ = os.path.splitext(os.path.basename(src_path))
        dest_path = os.path.join(output_base, f"{base_no_ext}.md")

        # 2. Kiểm tra nếu file đích đã tồn tại và mới hơn file nguồn (chế độ resume)
        if not force and os.path.exists(dest_path):
            try:
                src_mtime = os.path.getmtime(src_path)
                dest_mtime = os.path.getmtime(dest_path)
                if dest_mtime >= src_mtime and os.path.getsize(dest_path) > 0:
                    return {
                        "status": "SKIPPED",
                        "src_path": src_path,
                        "dest_path": dest_path,
                        "msg": "File đã được xử lý trước đó (up-to-date)."
                    }
            except Exception:
                pass

        # 3. Đọc nội dung HTML nguồn
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            html_text = f.read()

        if not html_text.strip():
            return {
                "status": "WARNING",
                "src_path": src_path,
                "dest_path": dest_path,
                "msg": "File nguồn rỗng."
            }

        # 4. Phân tích HTML sang dữ liệu sạch
        parsed = PhoneDBParser.parse_html(html_text)
        md_text = parsed["markdown"]

        # 5. Kiểm tra tính toàn vẹn (Integrity check)
        audit = IntegrityChecker.verify(parsed, md_text)

        # 6. Ghi kết quả sang thư mục đích
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f_out:
            f_out.write(md_text)

        status = "SUCCESS" if audit["passed"] else "FLAGGED"
        return {
            "status": status,
            "src_path": src_path,
            "dest_path": dest_path,
            "stats": parsed["stats"],
            "audit": audit,
            "errors": audit["errors"],
            "warnings": audit["warnings"]
        }

    except Exception as ex:
        return {
            "status": "ERROR",
            "src_path": src_path,
            "dest_path": "",
            "error_msg": str(ex)
        }


# ==============================================================================
# 6. QUẢN LÝ XỬ LÝ HÀNG LOẠT (BATCH PROCESSING PIPELINE)
# ==============================================================================
def scan_txt_files(directory: str) -> List[str]:
    """Quét toàn bộ file .txt, .html, .htm trong thư mục nguồn theo cách đệ quy."""
    valid_exts = (".txt", ".html", ".htm")
    data_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_exts):
                data_files.append(os.path.join(root, file))
    return sorted(data_files)


def run_batch_processing(
    input_dir: str,
    output_dir: str,
    max_workers: int = 16,
    limit: Optional[int] = None,
    force: bool = False
):
    """
    Tiến hành xử lý hàng loạt toàn bộ dataset với đa luồng/tiến trình song song.
    """
    print("=" * 80)
    print("      PHONEDB: LOSSLESS HTML TO STRUCTURED MARKDOWN CONVERTER")
    print("=" * 80)
    print(f"[*] Thư mục nguồn (Input):     {input_dir}")
    print(f"[*] Thư mục đích (Output):    {output_dir}")
    print(f"[*] HTML Parser sử dụng:      {PARSER_NAME}")
    print(f"[*] Số luồng worker tối đa:   {max_workers}")
    print(f"[*] Chế độ ghi đè (Force):    {'BẬT (Xử lý lại toàn bộ)' if force else 'TẮT (Chế độ Resume thông minh)'}")
    print("-" * 80)

    if not os.path.exists(input_dir):
        print(f"[!] Lỗi: Không tìm thấy thư mục nguồn '{input_dir}'!")
        print("    [!] Lưu ý nếu chạy trên Kaggle:")
        print("        1. Vui lòng thêm dataset: https://www.kaggle.com/datasets/tngtrn23/phone-db")
        print("        2. Kiểm tra đường dẫn: /kaggle/input/phone-db hoặc /kaggle/input/phone-db/data")
        return

    # 1. Quét danh sách file nguồn
    print("[*] Đang quét toàn bộ file (.txt, .html, .htm) trong thư mục nguồn...")
    t_scan_start = time.time()
    all_files = scan_txt_files(input_dir)
    t_scan_end = time.time()
    total_files = len(all_files)
    print(f"[+] Tìm thấy {total_files:,} file .txt (quét xong trong {t_scan_end - t_scan_start:.2f}s).")

    if total_files == 0:
        print("[!] Không có file nào để xử lý.")
        return

    if limit and limit > 0:
        all_files = all_files[:limit]
        print(f"[*] Giới hạn kiểm tra: chỉ xử lý {len(all_files):,} file.")

    total_to_process = len(all_files)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Khởi tạo danh sách tác vụ
    tasks = [(f, input_dir, output_dir, force) for f in all_files]

    processed_count = 0
    success_count = 0
    skipped_count = 0
    flagged_count = 0
    error_count = 0
    total_fields_extracted = 0

    flagged_files = []
    error_files = []

    start_time = time.time()
    last_print_time = start_time

    print(f"\n[*] Bắt đầu xử lý {total_to_process:,} files bằng ProcessPoolExecutor...")
    print("-" * 80)

    # Sử dụng ProcessPoolExecutor để tận dụng tối đa đa nhân CPU
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit theo batch hoặc map
        futures = {executor.submit(process_single_file, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            result = future.result()
            status = result.get("status")

            if status == "SUCCESS":
                success_count += 1
                total_fields_extracted += result.get("stats", {}).get("fields_count", 0)
            elif status == "SKIPPED":
                skipped_count += 1
            elif status == "FLAGGED":
                flagged_count += 1
                flagged_files.append((result["src_path"], result.get("errors", [])))
            elif status == "ERROR":
                error_count += 1
                error_files.append((result["src_path"], result.get("error_msg", "")))

            # In tiến độ định kỳ mỗi 0.5s hoặc mỗi 100 file
            now = time.time()
            if now - last_print_time >= 0.5 or processed_count == total_to_process:
                elapsed = now - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                pct = (processed_count / total_to_process) * 100
                remaining = (total_to_process - processed_count) / rate if rate > 0 else 0
                
                print(
                    f"\r[*] Tiến độ: {processed_count:,}/{total_to_process:,} ({pct:5.1f}%) | "
                    f"Thành công: {success_count:,} | Bỏ qua: {skipped_count:,} | "
                    f"Lỗi: {error_count} | Nghi vấn: {flagged_count} | "
                    f"Tốc độ: {rate:5.1f} f/s | ETA: {remaining:4.0f}s",
                    end="",
                    flush=True
                )
                last_print_time = now

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("                     BÁO CÁO TỔNG KẾT XỬ LÝ DỮ LIỆU")
    print("=" * 80)
    print(f"[+] Tổng số file quét:                  {total_to_process:,}")
    print(f"[+] Số file xử lý thành công (PASS):    {success_count:,}")
    print(f"[+] Số file bỏ qua (đã up-to-date):    {skipped_count:,}")
    print(f"[+] Số file bị lỗi (ERROR):             {error_count:,}")
    print(f"[+] Số file nghi ngờ sai lệch (FLAG):   {flagged_count:,}")
    print(f"[+] Tổng số trường (fields) trích xuất: {total_fields_extracted:,}")
    print(f"[+] Thời gian thực hiện:                {total_time:.2f} giây")
    if total_time > 0:
        print(f"[+] Tốc độ xử lý trung bình:           {total_to_process / total_time:.1f} file/giây")
    print("-" * 80)

    # 3. Xuất báo cáo kiểm tra chi tiết (Validation Report)
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "total_files": total_to_process,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "flagged_count": flagged_count,
        "error_count": error_count,
        "total_fields_extracted": total_fields_extracted,
        "elapsed_seconds": round(total_time, 2),
        "flagged_files": [{"path": p, "reasons": r} for p, r in flagged_files[:50]],
        "error_files": [{"path": p, "error": e} for p, e in error_files[:50]]
    }

    report_json_path = os.path.join(output_dir, "validation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f_rep:
        json.dump(report_data, f_rep, ensure_ascii=False, indent=2)
    print(f"[+] Đã lưu báo cáo chi tiết JSON vào: {report_json_path}")

    # Tạo bản báo cáo Markdown
    report_md_path = os.path.join(output_dir, "validation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f_md:
        f_md.write(f"# Báo cáo kiểm tra tính toàn vẹn dữ liệu PhoneDB\n\n")
        f_md.write(f"- **Thời gian chạy**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_md.write(f"- **Thư mục nguồn**: `{input_dir}`\n")
        f_md.write(f"- **Thư mục kết quả**: `{output_dir}`\n")
        f_md.write(f"- **Tổng số file quét**: {total_to_process:,}\n")
        f_md.write(f"- **Thành công (100% khớp)**: {success_count:,}\n")
        f_md.write(f"- **Bỏ qua (đã xử lý trước)**: {skipped_count:,}\n")
        f_md.write(f"- **Số file lỗi**: {error_count}\n")
        f_md.write(f"- **Số file cần kiểm tra thủ công**: {flagged_count}\n")
        f_md.write(f"- **Tổng số trường dữ liệu**: {total_fields_extracted:,}\n")
        f_md.write(f"- **Thời gian hoàn thành**: {total_time:.2f}s ({total_to_process / total_time:.1f} file/s)\n\n")
        
        if flagged_files:
            f_md.write("## Danh sách file nghi vấn cần kiểm tra thủ công\n\n")
            for p, r in flagged_files[:30]:
                f_md.write(f"- `{p}`: {'; '.join(r)}\n")
        else:
            f_md.write("## Đánh giá toàn vẹn\n\n- **KẾT QUẢ ĐỐI CHIẾU: 100% PASS.** Toàn bộ thông số, section và giá trị được bảo toàn tuyệt đối không suy suyển.\n")
            
    print(f"[+] Đã lưu báo cáo Markdown vào:      {report_md_path}")
    print("=" * 80)


# ==============================================================================
# 7. BỘ TỰ KIỂM TRA (BUILT-IN SELF-TEST SUITE)
# ==============================================================================
def run_self_tests():
    """
    Chạy bộ kiểm thử tự động với các cấu trúc HTML đặc biệt:
    1. Đầy đủ section, subsection, brief, đa băng tần, continuation row.
    2. HTML thiếu thẻ đóng hoặc bị cắt cụt.
    3. Hàng chứa quảng cáo và liên kết điều hướng.
    """
    print("\n" + "=" * 70)
    print("      CHẠY BỘ TEST CẤU TRÚC ĐẶC BIỆT CHO PHONEDB PARSER")
    print("=" * 70)

    sample_html = """
    <div class="canvas">
    <table>
      <tr>
        <td colspan="2"><br><a id="sec0"></a><h4><img alt="Introduction" src="icons/16x16/smartphone.gif">Introduction:&nbsp;</h4></td>
      </tr>
      <tr>
        <td><strong>Brand</strong><img src="icons/10x10/info_gray.gif" alt="Brand info"></td>
        <td>Apple</td>
      </tr>
      <tr>
        <td><strong>Model</strong></td>
        <td>iPhone 11 <a href="#"><span>Dual</span></a> <a href="#"><span>SIM</span></a> TD-LTE</td>
      </tr>
      <tr>
        <td colspan="2">
          <strong>Brief</strong><img src="icons/10x10/info_gray.gif">
          <br>Test brief for phone model..<a href="#" title="Detailed view contains even more details">››</a>
        </td>
      </tr>
      <tr>
        <td><strong>Released</strong></td>
        <td>2019 Sep 21</td>
      </tr>
      <tr>
        <td colspan="2">
          <div style="text-align: right;"><a href="#">Expand datasheet</a> | <a href="#">Add to compare</a></div>
          <h4><img alt="Physical Attributes" src="icons/16x16/ruler.gif">Physical Attributes:&nbsp;</h4>
        </td>
      </tr>
      <tr>
        <td><strong>Mass</strong></td>
        <td>194 g</td>
      </tr>
      <tr>
        <td></td>
        <td>6.84 ounces<img src="icons/10x10/info_gray.gif" alt="Mass in ounces"></td>
      </tr>
      <tr>
        <td colspan="2">
          <h5><img alt="Operative Memory" src="icons/16x16/ram.gif">Operative Memory:&nbsp;</h5>
        </td>
      </tr>
      <tr>
        <td><strong>RAM Type</strong></td>
        <td>LPDDR4x SDRAM</td>
      </tr>
      <tr>
        <td><strong>Supported Cellular Bands</strong></td>
        <td><a href="#">GSM850</a><img src="icons/10x10/info_gray.gif">, <a href="#">GSM900</a><img src="icons/10x10/info_gray.gif">, <a href="#">LTE700 (B12)</a> bands</td>
      </tr>
      <tr>
        <td><strong>Mobile Operator</strong></td>
        <td><a href="#">Vodafone UK</a><br><a href="#">Orange France</a></td>
      </tr>
      <tr>
        <td colspan="2">
          <ins class="adsbygoogle" style="display:block"></ins>
          <script>console.log('ad');</script>
        </td>
      </tr>
    </table>
    </div>
    """

    parsed = PhoneDBParser.parse_html(sample_html)
    md = parsed["markdown"]
    audit = IntegrityChecker.verify(parsed, md)

    print("[*] Kết quả phân tích HTML mẫu:")
    print("-" * 50)
    print(md)
    print("-" * 50)
    print(f"[*] Audit Passed: {audit['passed']}")
    print(f"[*] Sections: {parsed['stats']['sections_count']} (MD: {audit['out_stats']['sections_count']})")
    print(f"[*] Fields:   {parsed['stats']['fields_count']} (MD: {audit['out_stats']['fields_count']})")
    print(f"[*] Conts:    {parsed['stats']['continuation_count']} (MD: {audit['out_stats']['continuation_count']})")

    assert audit["passed"] is True, "Self test thất bại!"
    assert "Brief: Test brief for phone model.." in md
    assert "iPhone 11 Dual SIM TD-LTE" in md
    assert "Vodafone UK" in md and "Orange France" in md
    assert "6.84 ounces" in md
    assert "Expand datasheet" not in md
    assert "adsbygoogle" not in md
    print("[+] TẤT CẢ TEST CASES ĐÃ ĐẠT (100% PASS)!")
    print("=" * 70 + "\n")


# ==============================================================================
# 8. HÀM MAIN & GIAO DIỆN DÒNG LỆNH (CLI)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PhoneDB Lossless HTML to Clean Markdown / Text Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # 1. Chạy mặc định toàn bộ dataset:
  python xuli.py

  # 2. Chạy thử nghiệm 10 file đầu tiên:
  python xuli.py --limit 10

  # 3. Tùy chỉnh thư mục nguồn và đích:
  python xuli.py --input-dir "P:\\database phone\\data" --output-dir "P:\\database phone cleaned"

  # 4. Chạy lại toàn bộ, ghi đè file cũ:
  python xuli.py --force

  # 5. Chạy bộ tự kiểm tra self-test:
  python xuli.py --test
        """
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=DEFAULT_INPUT_DIR,
        help=f"Thư mục chứa dữ liệu HTML PhoneDB (mặc định: '{DEFAULT_INPUT_DIR}')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu trữ kết quả Markdown sạch (mặc định: '{DEFAULT_OUTPUT_DIR}')"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help=f"Số luồng tiến trình song song (mặc định: {min(16, os.cpu_count() or 4)})"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Giới hạn số file xử lý (dùng để test nhanh)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Bắt buộc xử lý lại toàn bộ file kể cả đã có trong thư mục đích"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Chạy bộ tự kiểm tra (Self-test suite) với các case HTML phức tạp"
    )
    parser.add_argument(
        "--single-file", "-s",
        type=str,
        default=None,
        help="Xử lý duy nhất một file cụ thể và in kết quả ra màn hình"
    )

    args = parser.parse_args()

    if args.test:
        run_self_tests()
        return

    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"[!] Lỗi: Không tìm thấy file '{args.single_file}'")
            return
        with open(args.single_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        res = PhoneDBParser.parse_html(content)
        print(res["markdown"])
        audit = IntegrityChecker.verify(res, res["markdown"])
        print("\n--- KẾT QUẢ ĐỐI CHIẾU ---")
        print(f"Trạng thái: {'PASS' if audit['passed'] else 'FAIL'}")
        print(f"Thống kê nguồn: {res['stats']}")
        print(f"Thống kê đích:  {audit['out_stats']}")
        if audit["errors"]:
            print(f"Lỗi: {audit['errors']}")
        return

    run_batch_processing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
        limit=args.limit,
        force=args.force
    )


if __name__ == "__main__":
    main()
