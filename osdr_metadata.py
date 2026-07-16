# osdr_metadata.py

import requests
from typing import Any

BASE_URL = "https://visualization.osdr.nasa.gov/biodata/api/v2"


class OSDRMetadataError(Exception):
    pass


def fetch_dataset_metadata(dataset_id: str, timeout: int = 30) -> dict[str, Any]:
    dataset_id = dataset_id.upper().strip()
    url = f"{BASE_URL}/dataset/{dataset_id}/"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise OSDRMetadataError(
            f"Failed to fetch metadata for {dataset_id}: {e}"
        ) from e


def _as_text(value: Any) -> str | None:
    """
    Convert OSDR metadata fields into readable text.
    Handles strings, lists, and missing values.
    """
    if value is None or value == "":
        return None

    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value if item)

    return str(value)


def get_study_summary(dataset_id: str) -> dict[str, str | None]:
    """
    Return the main text fields from an OSDR study.

    Includes:
    - study protocol description
    - study publication title
    - study description
    """
    dataset_id = dataset_id.upper().strip()
    data = fetch_dataset_metadata(dataset_id)

    metadata = (
        data.get(dataset_id, {})
        .get("metadata", {})
    )

    return {
        "dataset_id": dataset_id,
        "study_title": _as_text(metadata.get("study title")),
        "study_description": _as_text(metadata.get("study description")),
        "study_publication_title": _as_text(metadata.get("study publication title")),
        "study_protocol_description": _as_text(metadata.get("study protocol description")),
    }


def format_study_summary(summary: dict[str, str | None]) -> str:
    """
    Format study summary dictionary into a readable text block.
    """
    sections = []

    labels = {
        "dataset_id": "Dataset ID",
        "study_title": "Study Title",
        "study_description": "Study Description",
        "study_publication_title": "Study Publication Title",
        "study_protocol_description": "Study Protocol Description",
    }

    for key, label in labels.items():
        value = summary.get(key)
        if value:
            sections.append(f"{label}\n{'-' * len(label)}\n{value}")

    return "\n\n".join(sections)


if __name__ == "__main__":
    dataset = "OSD-48"

    summary = get_study_summary(dataset)
    print(format_study_summary(summary))