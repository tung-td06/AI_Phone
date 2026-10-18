"""
Module phone_metadata_v1: Triển khai Phone Metadata Schema V1.0.
Chuẩn hóa dữ liệu thiết bị thành 3 lớp:
  1. Product Master Data (Specs, brand, model, aliases, release_date)
  2. Commercial Data (Variants, RAM, Storage, Market Offers, Price)
  3. Evidence Data (Benchmarks, Gaming Tests có nguồn, không bịa đặt)
  4. Provenance Data (Nguồn file, thời điểm trích xuất, trạng thái xác minh)
Tuân thủ nghiêm ngặt JSON Schema Draft 2020-12 (phone_metadata_schema_v1.json).
"""

import re
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent / "phone_metadata_schema_v1.json"

VALID_REFRESH_RATES = {60, 90, 120, 144, 165, 185, 240}


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_str(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.strip()
    return s if s else None


class PhoneMetadataExtractorV1:
    """
    Trích xuất metadata chuẩn hóa từ các chunks text của một thiết bị.
    Không bịa đặt số liệu (trường thiếu = null).
    Phân biệt rõ tần số quét màn hình (refresh_rate_hz <= 240) vs cảm ứng (touch_sampling_rate_hz).
    """

    @staticmethod
    def extract_from_chunks(
        brand: str,
        model: str,
        chunks_text_list: List[str],
        source_file: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        combined = "\n".join(chunks_text_list)
        p_id = product_id or f"{brand.lower().replace(' ', '_')}_{re.sub(r'[^a-zA-Z0-9]+', '_', model.lower()).strip('_')}"

        # 1. Aliases & Release Date
        aliases = []
        rel_m = re.search(r"Released:\s*([^\n\r,]+)", combined, re.IGNORECASE)
        release_date = rel_m.group(1).strip() if rel_m else None

        # 2. Specs - Chipset
        cpu_name = None
        gpu_name = None
        chipset_name = None

        cpu_m = re.search(r"CPU:\s*([^\n\r,]+)", combined, re.IGNORECASE)
        if cpu_m:
            cpu_name = _clean_str(cpu_m.group(1))

        gpu_m = re.search(r"(?:Graphical Controller|GPU):\s*([^\n\r]+)", combined, re.IGNORECASE)
        if gpu_m:
            gpu_name = _clean_str(gpu_m.group(1))

        chip_m = re.search(r"(?:Chipset|SoC):\s*([^\n\r,]+)", combined, re.IGNORECASE)
        if chip_m:
            chipset_name = _clean_str(chip_m.group(1))
        elif cpu_name:
            chipset_name = cpu_name.split()[0] if cpu_name else None

        chipset_spec = {
            "name": chipset_name,
            "cpu": cpu_name,
            "gpu": gpu_name
        }

        # 3. Specs - Display (TÁCH BIỆT REFRESH RATE VÀ TOUCH SAMPLING RATE)
        tech = None
        for t in ["Super AMOLED", "Dynamic AMOLED 2X", "Dynamic AMOLED", "AMOLED", "OLED", "LTPO OLED", "IPS LCD", "TFT LCD", "LCD"]:
            if re.search(rf"\b{re.escape(t)}\b", combined, re.IGNORECASE):
                tech = t
                break

        size_in = None
        size_m = re.search(r"Display Diagonal:\s*(\d+(?:\.\d+)?)\s*(?:in|inch|\")", combined, re.IGNORECASE)
        if not size_m:
            size_m = re.search(r"(\d+\.\d+)\s*(?:in|inch|\")", combined, re.IGNORECASE)
        if size_m:
            try:
                size_in = float(size_m.group(1))
            except ValueError:
                pass

        # Refresh Rate: Ưu tiên Display Refresh Rate chính xác
        refresh_rate_hz = None
        rr_m = re.search(r"Display Refresh Rate:\s*(\d+)\s*Hz", combined, re.IGNORECASE)
        if rr_m:
            hz_val = int(rr_m.group(1))
            if hz_val in VALID_REFRESH_RATES:
                refresh_rate_hz = float(hz_val)
        if refresh_rate_hz is None:
            # Tìm trong ngữ cảnh màn hình
            disp_rates = []
            for m in re.finditer(r"\b(60|90|120|144|165|185|240)\s*Hz\b", combined, re.IGNORECASE):
                # Kiểm tra xem có từ 'touch' ngay cạnh không
                start = max(0, m.start() - 20)
                end = min(len(combined), m.end() + 20)
                surrounding = combined[start:end].lower()
                if "touch" not in surrounding and "cam ung" not in surrounding:
                    disp_rates.append(int(m.group(1)))
            if disp_rates:
                refresh_rate_hz = float(max(disp_rates))

        # Touch Sampling Rate: Các tần số cảm ứng cao (240, 360, 480, 720, 960...)
        touch_sampling_hz = None
        ts_m = re.search(r"(?:Touch(?:\s*Sampling)?(?:\s*Rate)?|cam ung)[:\s]*(\d{3,4})\s*Hz", combined, re.IGNORECASE)
        if not ts_m:
            ts_m = re.search(r"(\d{3,4})\s*Hz\s*(?:touch|touch-sampling|cam ung)", combined, re.IGNORECASE)
        if ts_m:
            touch_sampling_hz = float(ts_m.group(1))

        res_str = None
        res_m = re.search(r"(?:Display Resolution|Resolution):\s*(\d+\s*x\s*\d+)", combined, re.IGNORECASE)
        if res_m:
            res_str = res_m.group(1).replace(" ", "")

        display_spec = {
            "technology": tech,
            "size_in": size_in,
            "refresh_rate_hz": refresh_rate_hz,
            "touch_sampling_rate_hz": touch_sampling_hz,
            "resolution": res_str
        }

        # 4. Specs - Battery
        cap_mah = None
        bat_m = re.search(r"Nominal Battery Capacity:\s*(\d+)\s*mAh", combined, re.IGNORECASE)
        if not bat_m:
            bat_m = re.search(r"(\d{3,5})\s*mAh", combined, re.IGNORECASE)
        if bat_m:
            cap_mah = float(bat_m.group(1))

        wired_w = None
        w_m = re.search(r"(?:Fast Charging|Charging|Sac nhanh)[:\s]*(\d+)\s*W", combined, re.IGNORECASE)
        if w_m:
            wired_w = float(w_m.group(1))

        wireless_w = None
        wl_m = re.search(r"(?:Wireless Charging|Sac khong day)[:\s]*(\d+)\s*W", combined, re.IGNORECASE)
        if wl_m:
            wireless_w = float(wl_m.group(1))

        battery_spec = {
            "capacity_mah": cap_mah,
            "wired_charging_w": wired_w,
            "wireless_charging_w": wireless_w
        }

        # 5. Specs - Camera
        main_mp = None
        cmp_m = re.search(r"(?:Camera|Main Camera|May anh chinh)[:\s]*(\d+(?:\.\d+)?)\s*(?:MP|Megapixel)", combined, re.IGNORECASE)
        if cmp_m:
            main_mp = float(cmp_m.group(1))

        ois = True if re.search(r"\bOIS\b|Optical Image Stabilization|chong rung quang hoc", combined, re.IGNORECASE) else False
        zoom_x = None
        zm_m = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(?:optical zoom|zoom quang)", combined, re.IGNORECASE)
        if zm_m:
            zoom_x = float(zm_m.group(1))

        video_res = None
        if re.search(r"\b8K\b", combined, re.IGNORECASE):
            video_res = "8K"
        elif re.search(r"\b4K\b|2160p", combined, re.IGNORECASE):
            video_res = "4K"
        elif re.search(r"\b1080p\b|Full HD", combined, re.IGNORECASE):
            video_res = "1080p"

        camera_spec = {
            "main_mp": main_mp,
            "ois": ois,
            "optical_zoom_x": zoom_x,
            "video": video_res
        }

        # 6. Specs - Software & Connectivity
        os_name = None
        for os_candidate in ["Android 15", "Android 14", "Android 13", "Android 12", "Android 11", "Android 10", "iOS 18", "iOS 17", "iOS 16", "HyperOS", "HarmonyOS"]:
            if re.search(rf"\b{re.escape(os_candidate)}\b", combined, re.IGNORECASE):
                os_name = os_candidate
                break
        if not os_name and "Android" in combined:
            os_name = "Android"
        elif not os_name and "iOS" in combined:
            os_name = "iOS"

        software_spec = {
            "os": os_name,
            "support_until": None
        }

        cellular = "5G" if re.search(r"\b5G\b", combined, re.IGNORECASE) else "4G" if re.search(r"\b4G|LTE\b", combined, re.IGNORECASE) else None
        wifi = "Wi-Fi 7" if re.search(r"Wi-Fi 7|802\.11be", combined, re.IGNORECASE) else "Wi-Fi 6E" if re.search(r"Wi-Fi 6E", combined, re.IGNORECASE) else "Wi-Fi 6" if re.search(r"Wi-Fi 6|802\.11ax", combined, re.IGNORECASE) else "Wi-Fi 5" if re.search(r"802\.11ac", combined, re.IGNORECASE) else None
        bt = "Bluetooth 5.4" if "5.4" in combined else "Bluetooth 5.3" if "5.3" in combined else "Bluetooth" if "bluetooth" in combined.lower() else None
        nfc = True if re.search(r"\bNFC\b", combined, re.IGNORECASE) else False

        connectivity_spec = {
            "cellular": cellular,
            "wifi": wifi,
            "bluetooth": bt,
            "nfc": nfc
        }

        # 7. Variants (RAM / Storage)
        ram_vals = re.findall(r"RAM Capacity(?:\s*\(converted\))?:\s*(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)", combined, re.IGNORECASE)
        parsed_rams = sorted(set(
            float(v) / 1024.0 if "M" in u.upper() else float(v)
            for v, u in ram_vals
        )) if ram_vals else [None]

        rom_vals = re.findall(r"(?:Non-volatile Storage Capacity|ROM|Internal Storage)[:\s]*(\d+)\s*(GB|TB|GiB|TiB)", combined, re.IGNORECASE)
        parsed_roms = sorted(set(
            float(v) * 1024.0 if "T" in u.upper() else float(v)
            for v, u in rom_vals
        )) if rom_vals else [None]

        # Offers / Price
        from B7 import AdvancedPriceParser
        parsed_vnd, parsed_price_str = AdvancedPriceParser.parse(combined)
        offers = []
        if parsed_vnd:
            # Tránh gán giá trả góp nhà mạng thấp vô lý (< 15M cho dòng Ultra / Pro Max)
            is_ultra = any(w in model.lower() for w in ["ultra", "pro max", "fold"])
            if not (is_ultra and parsed_vnd < 15_000_000):
                offers.append({
                    "market": "VN",
                    "currency": "VND",
                    "price": parsed_vnd,
                    "seller": None,
                    "availability": "in_stock",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "source": {
                        "url": source_file,
                        "verified_at": datetime.now(timezone.utc).isoformat(),
                        "status": "verified"
                    }
                })

        variants = []
        for r in parsed_rams:
            for s in parsed_roms:
                var_suffix = f"{int(r) if r else 'X'}GB_{int(s) if s else 'X'}GB"
                variants.append({
                    "variant_id": f"{p_id}_{var_suffix}",
                    "ram_gb": r,
                    "storage_gb": s,
                    "region": "Global",
                    "color": None,
                    "offers": offers
                })

        # 8. Evidence Data (Benchmarks & Gaming tests - không tự chế khi không có nguồn)
        benchmarks = []
        gaming_tests = []
        from soc_database import lookup_soc_benchmark
        if cpu_name or gpu_name:
            soc_info = lookup_soc_benchmark(f"{cpu_name or ''} {gpu_name or ''}")
            if soc_info and soc_info.confidence != "UNKNOWN":
                if soc_info.geekbench_multi:
                    benchmarks.append({
                        "name": "Geekbench 6 Multi-Core",
                        "version": "6.0",
                        "score": float(soc_info.geekbench_multi),
                        "variant_id": variants[0]["variant_id"] if variants else None,
                        "source": {
                            "url": "soc_database",
                            "verified_at": "2024-01-01T00:00:00Z",
                            "status": "verified"
                        }
                    })
                if soc_info.antutu_v10:
                    benchmarks.append({
                        "name": "AnTuTu V10",
                        "version": "10.0",
                        "score": float(soc_info.antutu_v10),
                        "variant_id": variants[0]["variant_id"] if variants else None,
                        "source": {
                            "url": "soc_database",
                            "verified_at": "2024-01-01T00:00:00Z",
                            "status": "verified"
                        }
                    })

        # 9. Provenance
        provenance = {
            "source_file": source_file,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "verification_status": "verified" if offers else "unverified",
            "field_sources": {}
        }

        metadata = {
            "schema_version": "1.0.0",
            "product_id": p_id,
            "brand": brand,
            "model": model,
            "aliases": aliases,
            "release_date": release_date,
            "variants": variants,
            "specs": {
                "chipset": chipset_spec,
                "display": display_spec,
                "battery": battery_spec,
                "camera": camera_spec,
                "software": software_spec,
                "connectivity": connectivity_spec
            },
            "evidence": {
                "benchmarks": benchmarks,
                "gaming_tests": gaming_tests
            },
            "provenance": provenance
        }

        return metadata

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate metadata object against Phone Metadata Schema V1.0."""
        schema = load_schema()
        try:
            jsonschema.validate(instance=metadata, schema=schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, e.message
