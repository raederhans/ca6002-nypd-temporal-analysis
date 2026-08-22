"""Download and freeze the official NYPD Arrest Data (Year to Date) snapshot.

The downloader uses the official NYC Open Data Socrata API, paginates with the
stable total order ARREST_KEY plus Socrata ``:id``, and only promotes its unique
temporary CSV to the frozen snapshot when server counts and source revisions
before and after the download agree with the rows written. An optional
SOCRATA_APP_TOKEN is read from the environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import tempfile
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DATASET_ID = "uip8-fykc"
DATASET_NAME = "NYPD Arrest Data (Year to Date)"
SOURCE = "NYC Open Data / NYPD"
BASE_URL = "https://data.cityofnewyork.us"
DEFAULT_PAGE_SIZE = 25_000
SOCRATA_ROW_ID_ALIAS = "__socrata_row_id"
SOCRATA_TIE_BREAKER = ":id"
LOGGER = logging.getLogger("download_nypd_data")


class SnapshotDownloadError(RuntimeError):
    """Raised when a consistent full snapshot cannot be frozen."""


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json, text/csv;q=0.9, */*;q=0.8",
            "User-Agent": "CA6002-NYPD-Part1/1.0",
        }
    )
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    if app_token:
        session.headers["X-App-Token"] = app_token
        LOGGER.info("Using SOCRATA_APP_TOKEN from the environment.")
    else:
        LOGGER.info("No Socrata App Token found; continuing with anonymous access.")
    return session


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: tuple[int, int] = (15, 120),
) -> Any:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _server_count(session: requests.Session) -> int:
    payload = _get_json(
        session,
        f"{BASE_URL}/resource/{DATASET_ID}.json",
        params={"$select": "count(*)"},
    )
    if not payload or "count" not in payload[0]:
        raise SnapshotDownloadError(f"Unexpected count response: {payload!r}")
    return int(payload[0]["count"])


def _official_schema(session: requests.Session) -> tuple[dict[str, Any], str]:
    metadata = _get_json(session, f"{BASE_URL}/api/views/{DATASET_ID}")
    columns = metadata.get("columns", [])
    arrest_key = next(
        (
            column.get("fieldName")
            for column in columns
            if str(column.get("name", "")).upper() == "ARREST_KEY"
            or str(column.get("fieldName", "")).lower() == "arrest_key"
        ),
        None,
    )
    if not arrest_key:
        raise SnapshotDownloadError("Official metadata does not expose ARREST_KEY.")
    return metadata, str(arrest_key)


def _order_token(value: str) -> tuple[int, Decimal | str]:
    """Return a value that follows Socrata numeric ordering when applicable."""

    try:
        return (0, Decimal(value))
    except (InvalidOperation, ValueError):
        return (1, value)


def _new_part_path(raw_path: Path) -> Path:
    """Create a same-directory part file owned by this downloader invocation."""

    descriptor, name = tempfile.mkstemp(
        prefix=f".{raw_path.name}.", suffix=".part", dir=raw_path.parent
    )
    os.close(descriptor)
    return Path(name)


def _download_once(
    session: requests.Session,
    part_path: Path,
    *,
    expected_rows: int,
    order_field: str,
    page_size: int,
) -> dict[str, Any]:
    row_count = 0
    response_header: list[str] | None = None
    output_header: list[str] | None = None
    arrest_key_index: int | None = None
    row_id_index: int | None = None
    date_index: int | None = None
    key_counts: Counter[str] = Counter()
    exact_row_counts: Counter[tuple[str, ...]] = Counter()
    min_date: str | None = None
    max_date: str | None = None
    previous_primary_order: tuple[int, Decimal | str] | None = None
    socrata_row_ids: set[str] = set()
    page_boundaries: list[dict[str, Any]] = []
    order_clause = f"{order_field}, {SOCRATA_TIE_BREAKER}"
    select_clause = f"*, {SOCRATA_TIE_BREAKER} as {SOCRATA_ROW_ID_ALIAS}"

    with part_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        for offset in range(0, expected_rows, page_size):
            response = session.get(
                f"{BASE_URL}/resource/{DATASET_ID}.csv",
                params={
                    "$limit": page_size,
                    "$offset": offset,
                    "$order": order_clause,
                    "$select": select_clause,
                },
                timeout=(15, 180),
            )
            response.raise_for_status()
            rows = list(csv.reader(io.StringIO(response.text)))
            if not rows:
                raise SnapshotDownloadError(f"Empty CSV response at offset {offset}.")

            page_header, page_rows = rows[0], rows[1:]
            if response_header is None:
                response_header = page_header
                normalized_header = [name.strip().lower() for name in response_header]
                if order_field.lower() not in normalized_header:
                    raise SnapshotDownloadError(
                        f"Ordering field {order_field!r} missing from CSV header {response_header!r}."
                    )
                if normalized_header.count(SOCRATA_ROW_ID_ALIAS) != 1:
                    raise SnapshotDownloadError(
                        "Socrata row-id tie-breaker is absent or ambiguous in "
                        f"CSV header {response_header!r}."
                    )
                arrest_key_index = normalized_header.index(order_field.lower())
                row_id_index = normalized_header.index(SOCRATA_ROW_ID_ALIAS)
                date_index = normalized_header.index("arrest_date")
                output_header = [
                    name
                    for index, name in enumerate(response_header)
                    if index != row_id_index
                ]
                writer.writerow(output_header)
            elif page_header != response_header:
                raise SnapshotDownloadError(
                    f"Schema changed during pagination at offset {offset}."
                )

            if not page_rows:
                raise SnapshotDownloadError(
                    f"Unexpected empty page at offset {offset} before expected count."
                )
            assert arrest_key_index is not None
            assert row_id_index is not None
            assert date_index is not None

            first_key = page_rows[0][arrest_key_index]
            last_key = page_rows[-1][arrest_key_index]
            first_row_id = page_rows[0][row_id_index]
            last_row_id = page_rows[-1][row_id_index]
            page_boundaries.append(
                {
                    "offset": offset,
                    "rows": len(page_rows),
                    "first_arrest_key": first_key,
                    "last_arrest_key": last_key,
                    "first_socrata_row_id": first_row_id,
                    "last_socrata_row_id": last_row_id,
                }
            )

            for row in page_rows:
                if len(row) != len(response_header):
                    raise SnapshotDownloadError(
                        f"Row width {len(row)} does not match response header width "
                        f"{len(response_header)}."
                    )
                key = row[arrest_key_index]
                row_id = row[row_id_index]
                if not row_id or row_id in socrata_row_ids:
                    raise SnapshotDownloadError(
                        f"Socrata row-id tie-breaker is blank or duplicated: {row_id!r}."
                    )
                current_primary_order = _order_token(key)
                if (
                    previous_primary_order is not None
                    and current_primary_order < previous_primary_order
                ):
                    raise SnapshotDownloadError(
                        "Primary pagination order regressed: "
                        f"previous={previous_primary_order!r}, "
                        f"current={current_primary_order!r}."
                    )
                # Socrata :id is an opaque system value. Its uniqueness makes
                # the requested composite order total, but its returned string
                # representation is not assumed to expose Socrata's collation.
                previous_primary_order = current_primary_order
                socrata_row_ids.add(row_id)
                key_counts[key] += 1
                output_row = [
                    value for index, value in enumerate(row) if index != row_id_index
                ]
                exact_row_counts[tuple(output_row)] += 1
                raw_date = row[date_index].strip()
                if raw_date:
                    iso_date = raw_date[:10]
                    min_date = iso_date if min_date is None else min(min_date, iso_date)
                    max_date = iso_date if max_date is None else max(max_date, iso_date)
                writer.writerow(output_row)
            row_count += len(page_rows)
            LOGGER.info(
                "Downloaded %s/%s rows (offset=%s, page=%s).",
                f"{row_count:,}",
                f"{expected_rows:,}",
                f"{offset:,}",
                f"{len(page_rows):,}",
            )

    if response_header is None or output_header is None:
        raise SnapshotDownloadError("No CSV header was downloaded.")
    duplicate_key_rows = sum(count - 1 for count in key_counts.values() if count > 1)
    exact_duplicate_rows = sum(
        count - 1 for count in exact_row_counts.values() if count > 1
    )
    return {
        "row_count": row_count,
        "column_count": len(output_header),
        "columns": output_header,
        "min_arrest_date": min_date,
        "max_arrest_date": max_date,
        "duplicate_arrest_key_rows": duplicate_key_rows,
        "exact_duplicate_rows": exact_duplicate_rows,
        "page_count": len(page_boundaries),
        "page_boundaries": page_boundaries,
        "pagination_order_clause": order_clause,
        "pagination_tie_breaker_field": SOCRATA_TIE_BREAKER,
        "pagination_tie_breaker_unique": len(socrata_row_ids) == row_count,
        "pagination_order_is_total": len(socrata_row_ids) == row_count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _promote_snapshot_no_clobber(part_path: Path, raw_path: Path) -> None:
    """Atomically publish ``part_path`` only when ``raw_path`` is absent.

    Both paths are created in the same raw-data directory.  A hard-link create
    is atomic and fails when the destination already exists, unlike
    ``Path.replace`` which can overwrite a snapshot created by another process
    between an existence check and promotion.
    """

    try:
        os.link(part_path, raw_path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Frozen snapshot was created concurrently and was not overwritten: {raw_path}."
        ) from exc
    except OSError as exc:
        raise SnapshotDownloadError(
            "Unable to atomically promote the snapshot without overwrite; "
            f"the filesystem may not support same-volume hard links: {exc}"
        ) from exc
    try:
        part_path.unlink()
    except OSError as exc:
        LOGGER.warning(
            "Snapshot was frozen but temporary hard-link cleanup failed for %s: %s",
            part_path,
            exc,
        )


def download_snapshot(
    project_root: Path,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Download a count-reconciled official snapshot and return its metadata."""

    project_root = Path(project_root).resolve()
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs" / "part1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieved_at = datetime.now().astimezone()
    retrieval_date = retrieved_at.date().isoformat()
    raw_path = raw_dir / f"nypd_arrests_ytd_{retrieval_date}.csv"
    metadata_path = output_dir / "dataset_snapshot_metadata.json"
    if raw_path.exists():
        raise FileExistsError(
            f"Frozen snapshot already exists and will not be overwritten: {raw_path}."
        )

    session = _session()
    part_path = _new_part_path(raw_path)
    last_error: Exception | None = None
    attempt_details: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            metadata_before, order_field = _official_schema(session)
            revision_before = metadata_before.get("rowsUpdatedAt")
            expected_before = _server_count(session)
            LOGGER.info(
                "Attempt %s/%s: server reports %s rows.",
                attempt,
                max_attempts,
                f"{expected_before:,}",
            )
            stats = _download_once(
                session,
                part_path,
                expected_rows=expected_before,
                order_field=order_field,
                page_size=page_size,
            )
            expected_after = _server_count(session)
            metadata_after, order_field_after = _official_schema(session)
            revision_rechecked_at = datetime.now().astimezone().isoformat()
            revision_after = metadata_after.get("rowsUpdatedAt")
            revision_stable = (
                revision_before is not None
                and revision_before == revision_after
                and order_field == order_field_after
            )
            consistent = (
                stats["row_count"] == expected_before == expected_after
                and revision_stable
            )
            attempt_details.append(
                {
                    "attempt": attempt,
                    "expected_before": expected_before,
                    "downloaded": stats["row_count"],
                    "expected_after": expected_after,
                    "source_revision_before": revision_before,
                    "source_revision_after": revision_after,
                    "source_revision_stable": revision_stable,
                    "pagination_order_clause": stats[
                        "pagination_order_clause"
                    ],
                    "consistent": consistent,
                }
            )
            if not consistent:
                raise SnapshotDownloadError(
                    "Server count/revision changed or download was incomplete: "
                    f"before={expected_before}, downloaded={stats['row_count']}, "
                    f"after={expected_after}, revision_before={revision_before}, "
                    f"revision_after={revision_after}."
                )

            _promote_snapshot_no_clobber(part_path, raw_path)
            dataset_updated_at = metadata_after.get("rowsUpdatedAt")
            if dataset_updated_at:
                dataset_updated_at = datetime.fromtimestamp(
                    int(dataset_updated_at), tz=timezone.utc
                ).isoformat()
            snapshot_metadata: dict[str, Any] = {
                "dataset_name": metadata_after.get("name", DATASET_NAME),
                "dataset_id": DATASET_ID,
                "source": SOURCE,
                "dataset_page": f"{BASE_URL}/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/{DATASET_ID}/about_data",
                "retrieval_date": retrieval_date,
                "retrieved_at": retrieved_at.isoformat(),
                "dataset_updated_at": dataset_updated_at,
                "row_count": stats["row_count"],
                "column_count": stats["column_count"],
                "columns": stats["columns"],
                "min_arrest_date": stats["min_arrest_date"],
                "max_arrest_date": stats["max_arrest_date"],
                "api_method": (
                    "Official Socrata SODA CSV API; paginated with $limit/$offset "
                    f"and stable total $order={stats['pagination_order_clause']}"
                ),
                "api_endpoint": f"{BASE_URL}/resource/{DATASET_ID}.csv",
                "page_size": page_size,
                "page_count": stats["page_count"],
                "api_expected_rows_before": expected_before,
                "downloaded_rows": stats["row_count"],
                "api_expected_rows_after": expected_after,
                "count_match": True,
                "source_revision_before": revision_before,
                "source_revision_after": revision_after,
                "source_revision_match": True,
                "source_revision_verification_method": (
                    "Official metadata and row count queried immediately before and "
                    "after paginated download; no source revision change was observed."
                ),
                "source_revision_rechecked_at": revision_rechecked_at,
                "api_row_count_rechecked_after_download": expected_after,
                "pagination_order_field": order_field,
                "pagination_order_clause": stats["pagination_order_clause"],
                "pagination_tie_breaker_field": stats[
                    "pagination_tie_breaker_field"
                ],
                "pagination_tie_breaker_unique": stats[
                    "pagination_tie_breaker_unique"
                ],
                "pagination_primary_key_unique": stats[
                    "duplicate_arrest_key_rows"
                ]
                == 0,
                "pagination_order_is_total": stats[
                    "pagination_order_is_total"
                ],
                "pagination_order_monotonic": True,
                "duplicate_arrest_key_rows_seen_during_download": stats[
                    "duplicate_arrest_key_rows"
                ],
                "exact_duplicate_rows_seen_during_download": stats[
                    "exact_duplicate_rows"
                ],
                "raw_file": raw_path.relative_to(project_root).as_posix(),
                "raw_file_sha256": _sha256(raw_path),
                "attempts": attempt_details,
                "pages": stats["page_boundaries"],
                "app_token_used": bool(os.environ.get("SOCRATA_APP_TOKEN")),
            }
            metadata_path.write_text(
                json.dumps(snapshot_metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            LOGGER.info("Frozen snapshot: %s", raw_path)
            LOGGER.info("Snapshot metadata: %s", metadata_path)
            return snapshot_metadata
        except FileExistsError:
            if part_path.exists():
                part_path.unlink()
            raise
        except (requests.RequestException, OSError, ValueError, SnapshotDownloadError) as exc:
            last_error = exc
            LOGGER.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 8))

    if part_path.exists():
        part_path.unlink()
    raise SnapshotDownloadError(
        f"Unable to freeze a consistent snapshot after {max_attempts} attempts: {last_error}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (default: parent of src).",
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.page_size <= 0:
        raise SystemExit("--page-size must be positive.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    snapshot = download_snapshot(
        args.project_root,
        page_size=args.page_size,
        max_attempts=args.max_attempts,
    )
    print(
        json.dumps(
            {
                "raw_file": snapshot["raw_file"],
                "rows": snapshot["row_count"],
                "columns": snapshot["column_count"],
                "date_range": [
                    snapshot["min_arrest_date"],
                    snapshot["max_arrest_date"],
                ],
                "count_match": snapshot["count_match"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
