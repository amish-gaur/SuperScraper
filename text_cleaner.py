"""Regex-driven text cleaning helpers for scraped web data."""

from __future__ import annotations

import re


class TextCleaningUtility:
    """Normalize common semi-structured web text into typed ML-ready values."""

    @staticmethod
    def clean_price(price_str: str | object) -> float | None:
        if price_str is None:
            return None
        text = str(price_str).strip()
        if not text:
            return None

        text = text.replace("\u00a0", " ")
        match = re.search(r"[-+]?\d[\d.,\s]*", text)
        if not match:
            return None

        candidate = re.sub(r"\s+", "", match.group(0))
        if not candidate:
            return None

        has_comma = "," in candidate
        has_dot = "." in candidate
        if has_comma and has_dot:
            if candidate.rfind(",") > candidate.rfind("."):
                normalized = candidate.replace(".", "").replace(",", ".")
            else:
                normalized = candidate.replace(",", "")
        elif has_comma:
            comma_count = candidate.count(",")
            if comma_count == 1 and len(candidate.split(",")[-1]) <= 2:
                normalized = candidate.replace(",", ".")
            else:
                normalized = candidate.replace(",", "")
        else:
            normalized = candidate

        normalized = re.sub(r"[^0-9.\-+]", "", normalized)
        if normalized.count(".") > 1:
            head, tail = normalized.rsplit(".", 1)
            normalized = head.replace(".", "") + "." + tail
        try:
            return float(normalized)
        except ValueError:
            return None

    @staticmethod
    def extract_laptop_specs(spec_blob: str | object) -> dict[str, int | str]:
        if spec_blob is None:
            return {}
        text = str(spec_blob).strip()
        if not text:
            return {}

        extracted: dict[str, int | str] = {}

        ram_match = re.search(r"(\d+(?:\.\d+)?)\s*GB\s+(?:RAM|Memory)\b", text, flags=re.IGNORECASE)
        if ram_match:
            extracted["ram_gb"] = int(float(ram_match.group(1)))

        storage_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(TB|GB)\s+(?:SSD|HDD|Storage)\b",
            text,
            flags=re.IGNORECASE,
        )
        if storage_match:
            storage_value = float(storage_match.group(1))
            storage_unit = storage_match.group(2).upper()
            if storage_unit == "TB":
                storage_value *= 1024
            extracted["storage_gb"] = int(storage_value)

        gpu_patterns = (
            r"(NVIDIA\s+GeForce\s+[A-Za-z0-9\- ]+)",
            r"(AMD\s+Radeon\s+[A-Za-z0-9\- ]+)",
            r"(Intel\s+(?:Arc|Iris\s+Xe|UHD)\s+[A-Za-z0-9\- ]+)",
            r"(RTX\s+\d{3,4}(?:\s*Ti)?)",
            r"(GTX\s+\d{3,4}(?:\s*Ti)?)",
        )
        for pattern in gpu_patterns:
            gpu_match = re.search(pattern, text, flags=re.IGNORECASE)
            if gpu_match:
                extracted["gpu_model"] = gpu_match.group(1).strip(" ,")
                break

        return extracted
