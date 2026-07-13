from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


def _memory_paths(config_path: str | Path) -> dict[str, Path | int]:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = config["memory"]
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    root = resolve_from_file(memory["root_dir"], path)
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    return {
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "reflections": root / memory.get("reflection_memory_dir", "reflections"),
        "index": root / memory["index_path"],
        "max_chars": max_chars,
    }


def _read_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index


def _query_terms(query: str | None) -> list[str]:
    if not isinstance(query, str) or not query.strip():
        return []
    return [term.casefold() for term in re.split(r"\s+", query.strip()) if term.strip()]


def _recency_score(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400)
    except ValueError:
        return 0.0
    return round(max(0.0, 1.0 - min(age_days, 30.0) / 30.0), 4)


def _memory_score(metadata: dict, content: str, query: str | None, explicit: bool = False) -> tuple[float, str, dict]:
    terms = _query_terms(query)
    explicit_score = 100.0 if explicit else 0.0
    importance = float(metadata.get("importance", 0.0) or 0.0)
    access_count = int(metadata.get("access_count", 0) or 0)
    access_score = min(access_count, 10) * 0.05
    recency = _recency_score(metadata.get("last_accessed_at") or metadata.get("updated_at") or metadata.get("created_at"))
    if not terms:
        breakdown = {
            "explicit_score": explicit_score,
            "keyword_score": 0.0,
            "importance_score": importance,
            "recency_score": recency,
            "access_score": access_score,
        }
        total = sum(breakdown.values())
        reason = "explicit selection" if explicit else "global memory selection"
        return round(total, 4), reason, breakdown
    haystack = " ".join(
        str(metadata.get(key, ""))
        for key in ("title", "summary", "compact_summary", "memory_id", "memory_type", "conversation_id", "source")
    ).casefold()
    content_folded = content.casefold()
    keyword_hits = sum(haystack.count(term) * 2 + content_folded.count(term) for term in terms)
    keyword_score = float(keyword_hits)
    breakdown = {
        "explicit_score": explicit_score,
        "keyword_score": keyword_score,
        "importance_score": importance,
        "recency_score": recency,
        "access_score": access_score,
    }
    score = sum(breakdown.values())
    if keyword_hits:
        reason = "matched query keyword"
        if explicit:
            reason += " and explicit selection"
    else:
        reason = "selected by explicit id or global memory"
    return round(score, 4), reason, {key: round(value, 4) for key, value in breakdown.items()}


def _ensure_memory_metadata(memory_id: str, metadata: dict) -> dict:
    metadata.setdefault("importance", 0.5 if metadata.get("memory_type") == "global" else 0.3)
    metadata.setdefault("access_count", 0)
    metadata.setdefault("last_accessed_at", None)
    metadata.setdefault("tags", [])
    metadata.setdefault("source", metadata.get("memory_type", "conversation"))
    metadata.setdefault("memory_id", memory_id)
    return metadata


def _read_memory_document(paths: dict, metadata: dict) -> tuple[Path | None, str | None, dict | None]:
    relative_path = metadata.get("path")
    if not isinstance(relative_path, str):
        return None, None, {"type": "InvalidMetadata", "message": "memory path is missing"}
    document_path = (paths["root"] / relative_path).resolve()
    try:
        document_path.relative_to(paths["root"].resolve())
    except ValueError:
        return document_path, None, {"type": "InvalidPath", "message": "memory path escapes root"}
    if not document_path.is_file():
        return document_path, None, {"type": "FileNotFoundError", "message": f"memory file not found: {relative_path}"}
    return document_path, read_text(document_path), None


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    outdir: str | None = None,
) -> dict:
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    index_changed = False
    explicit_ids = list(dict.fromkeys(selected_memory_ids))
    candidate_ids = []
    candidate_reasons = {}
    for memory_id in explicit_ids:
        candidate_ids.append(memory_id)
        candidate_reasons[memory_id] = "explicit"
    if use_global_memory:
        for memory_id, item in index.items():
            if isinstance(item, dict) and item.get("memory_type") == "global":
                candidate_ids.append(memory_id)
                candidate_reasons.setdefault(memory_id, "global")
    query_terms = _query_terms(query)
    if query_terms:
        for memory_id, item in index.items():
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get(key, "")) for key in ("title", "summary", "compact_summary", "memory_id", "conversation_id", "source")).casefold()
            if any(term in text for term in query_terms):
                candidate_ids.append(memory_id)
                candidate_reasons.setdefault(memory_id, "query_index_match")
    candidate_ids = list(dict.fromkeys(candidate_ids))

    candidates = []
    errors = []
    for memory_id in candidate_ids:
        metadata = index.get(memory_id)
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        metadata = _ensure_memory_metadata(memory_id, metadata)
        index_changed = True
        document_path, original, read_error = _read_memory_document(paths, metadata)
        if read_error:
            errors.append({"memory_id": memory_id, **read_error})
            continue
        explicit = memory_id in explicit_ids
        score, selection_reason, score_breakdown = _memory_score(metadata, original, query, explicit)
        candidates.append(
            {
                "metadata": metadata,
                "memory_id": memory_id,
                "relative_path": metadata.get("path"),
                "original": original,
                "score": score,
                "score_breakdown": score_breakdown,
                "selection_reason": selection_reason,
                "candidate_reason": candidate_reasons.get(memory_id, "query_content_match"),
                "explicit": explicit,
            }
        )
    if query_terms:
        seen = {candidate["memory_id"] for candidate in candidates}
        for memory_id, metadata in index.items():
            if memory_id in seen or not isinstance(metadata, dict):
                continue
            metadata = _ensure_memory_metadata(memory_id, metadata)
            document_path, original, read_error = _read_memory_document(paths, metadata)
            if read_error:
                continue
            score, selection_reason, score_breakdown = _memory_score(metadata, original, query, False)
            if score_breakdown.get("keyword_score", 0) > 0:
                candidates.append(
                    {
                        "metadata": metadata,
                        "memory_id": memory_id,
                        "relative_path": metadata.get("path"),
                        "original": original,
                        "score": score,
                        "score_breakdown": score_breakdown,
                        "selection_reason": selection_reason,
                        "candidate_reason": "query_content_match",
                        "explicit": False,
                    }
                )
                seen.add(memory_id)
                index_changed = True
    candidates.sort(key=lambda item: (not item["explicit"], -float(item["score"]), item["memory_id"]))
    docs = []
    remaining = int(paths["max_chars"])
    any_truncated = False
    for candidate in candidates:
        metadata = candidate["metadata"]
        original = candidate["original"]
        included = original[:remaining] if remaining > 0 else ""
        truncated = len(included) < len(original)
        any_truncated = any_truncated or truncated
        if included:
            docs.append(
                {
                    "memory_id": candidate["memory_id"],
                    "memory_type": metadata.get("memory_type"),
                    "title": metadata.get("title", candidate["memory_id"]),
                    "path": candidate["relative_path"],
                    "score": candidate["score"],
                    "score_breakdown": candidate["score_breakdown"],
                    "selection_reason": candidate["selection_reason"],
                    "candidate_reason": candidate["candidate_reason"],
                    "importance": metadata.get("importance", 0.0),
                    "access_count": metadata.get("access_count", 0),
                    "content": included,
                    "original_chars": len(original),
                    "included_chars": len(included),
                    "truncated": truncated,
                }
            )
            remaining -= len(included)
            metadata["access_count"] = int(metadata.get("access_count", 0) or 0) + 1
            metadata["last_accessed_at"] = now_iso()
            index_changed = True
    if errors and docs:
        status = "partial"
    elif errors:
        status = "error"
    else:
        status = "success"
    result = {
        "status": status,
        "query": query,
        "retrieval_mode": "explicit_global_keyword" if query_terms else "explicit_global",
        "candidate_count": len(candidates),
        "selected_memory_docs": docs,
        "max_memory_chars": paths["max_chars"],
        "total_chars": sum(item["included_chars"] for item in docs),
        "truncated": any_truncated,
        "errors": errors,
    }
    if index_changed:
        write_json(index, paths["index"])
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "load",
                "status": status,
                "selected_ids": [item["memory_id"] for item in docs],
                "errors": errors,
            },
            output_dir / "memory_log.jsonl",
        )
    return result


def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict:
    conversation_id = _safe_conversation_id(conversation_id)
    if save_type not in {"conversation", "global", "reflection"}:
        raise ValueError("save_type must be conversation, global, or reflection")
    paths = _memory_paths(config_path)
    messages = read_json(messages_path)
    trace = read_json(trace_path)
    answer = read_text(answer_path).strip()
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    now = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    if save_type == "conversation":
        target_dir = paths["conversations"]
        relative_dir = "conversations"
    elif save_type == "global":
        target_dir = paths["global"]
        relative_dir = "global"
    else:
        target_dir = paths["reflections"]
        relative_dir = "reflections"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    title = f"{save_type.title()} {conversation_id}"
    tool_errors = []
    for turn in trace.get("turns", []):
        for message in turn.get("tool_messages", []):
            if message.get("status") == "error":
                tool_errors.append(f"{message.get('name', 'unknown')}:{message.get('tool_call_id', '')}")
    compact_parts = [
        f"answer={answer[:120]}",
        f"termination={trace.get('termination_reason', trace.get('status'))}",
    ]
    if tool_errors:
        compact_parts.append(f"tool_errors={', '.join(tool_errors[:3])}")
    summary = " | ".join(compact_parts)[:300]
    markdown = (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- created_or_updated_at: `{now}`\n\n"
        "## Final Answer\n\n"
        f"{answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"
    )
    write_text(markdown, target_path)
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    created_at = existing.get("created_at", now)
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": created_at,
        "updated_at": now,
        "importance": existing.get("importance", 0.4 if save_type == "conversation" else (0.6 if save_type == "global" else 0.5)),
        "last_accessed_at": existing.get("last_accessed_at"),
        "access_count": existing.get("access_count", 0),
        "tags": existing.get("tags", []),
        "source": existing.get("source", save_type),
        "compact_summary": summary,
    }
    write_json(index, paths["index"])
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": now,
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "saved_memory.json")
        append_jsonl(
            {"timestamp": now, "operation": "save", "status": "success", "memory_id": memory_id},
            output_dir / "memory_log.jsonl",
        )
    return result


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or save local memory documents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--select_memory_ids", nargs="*")
    parser.add_argument("--use_global_memory", type=parse_bool)
    parser.add_argument("--query")
    parser.add_argument("--save_type", choices=["conversation", "global", "reflection"])
    parser.add_argument("--save_input_path")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            base = input_path.parent
            result = save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            print(outdir / "saved_memory.json")
        else:
            if args.select_memory_ids is None and args.use_global_memory is None:
                raise ValueError("select mode requires --select_memory_ids or --use_global_memory")
            result = load_memory(
                str(config_path),
                args.select_memory_ids or [],
                bool(args.use_global_memory),
                args.query,
                str(outdir),
            )
            print(outdir / "selected_memory.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
