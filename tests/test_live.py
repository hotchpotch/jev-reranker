"""Semantic ordering checks against the real service; run with pytest --live.

Each fixture has four relevance levels: complete answer, partial answer, a
related overview without instructions, and an unrelated passage. The expected
order is fixed before execution; simply returning descending scores is not enough.
"""

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest

from jev_reranker import JevError, JevReranker

CASES = {
    "en": (
        "How do I enable automatic backups in AtlasDB, and which command restores the latest backup?",
        [
            (
                "To enable automatic backups in AtlasDB, set auto_backup=true in atlas.conf. "
                "To restore the latest backup, run atlas restore --latest."
            ),
            "To enable automatic backups in AtlasDB, set auto_backup=true in atlas.conf.",
            "AtlasDB supports automatic backups and recovery. See the backup settings guide for instructions.",
            "Chocolate cake is made with flour, eggs, sugar, and cocoa powder.",
        ],
    ),
    "ja": (
        "AtlasDB の自動バックアップを有効にする設定と、最新のバックアップを復元するコマンドは？",
        [
            (
                "AtlasDB の自動バックアップは atlas.conf に auto_backup=true を設定すると有効になります。"
                "最新のバックアップは atlas restore --latest で復元します。"
            ),
            "AtlasDB の自動バックアップは atlas.conf に auto_backup=true を設定すると有効になります。",
            "AtlasDB は自動バックアップと復元に対応しています。操作方法はバックアップの設定ガイドを参照してください。",
            "チョコレートケーキには小麦粉、卵、砂糖、ココアを使います。",
        ],
    ),
    "zh": (
        "如何启用 AtlasDB 的自动备份，使用什么命令恢复最新备份？",
        [
            "在 atlas.conf 中设置 auto_backup=true 即可启用 AtlasDB 自动备份。运行 atlas restore --latest 恢复最新备份。",
            "在 atlas.conf 中设置 auto_backup=true 即可启用 AtlasDB 自动备份。",
            "AtlasDB 支持自动备份和恢复。具体操作请参阅备份设置指南。",
            "巧克力蛋糕的原料包括面粉、鸡蛋、糖和可可粉。",
        ],
    ),
    "es": (
        "¿Cómo activo las copias de seguridad automáticas en AtlasDB y qué comando restaura la última copia?",
        [
            (
                "Para activar las copias automáticas de AtlasDB, configura auto_backup=true en atlas.conf. "
                "Para restaurar la última copia, ejecuta atlas restore --latest."
            ),
            "Para activar las copias automáticas de AtlasDB, configura auto_backup=true en atlas.conf.",
            "AtlasDB admite copias automáticas y recuperación. Consulta la guía de configuración de copias para las instrucciones.",
            "La tarta de chocolate lleva harina, huevos, azúcar y cacao.",
        ],
    ),
    "mixed": (
        "AtlasDB の自動バックアップを有効にする設定と、最新のバックアップを復元するコマンドは？",
        [
            (
                "To enable automatic backups in AtlasDB, set auto_backup=true in atlas.conf. "
                "To restore the latest backup, run atlas restore --latest."
            ),
            "在 atlas.conf 中设置 auto_backup=true 即可启用 AtlasDB 自动备份。",
            "AtlasDB admite copias automáticas y recuperación. Consulta la guía de configuración de copias para las instrucciones.",
            "チョコレートケーキには小麦粉、卵、砂糖、ココアを使います。",
        ],
    ),
}


@pytest.mark.live
@pytest.mark.parametrize("mode", ["listwise", "pointwise", "pairwise"])
@pytest.mark.parametrize("language", CASES)
def test_multilingual_order(language, mode):
    query, by_relevance = CASES[language]
    # Fixed shuffle prevents an implementation returning input order from passing.
    documents = [by_relevance[i] for i in [1, 3, 0, 2]]
    output = Path(".live-results")
    output.mkdir(exist_ok=True)
    path = output / f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}-{language}-{mode}.json"
    with JevReranker(mode=mode, max_retries=2) as ranker:
        try:
            raw = ranker.raw_rerank(query, documents, detail=True)
        except JevError as exc:
            path.write_text(json.dumps(exc.detail, ensure_ascii=False, indent=2) + "\n")
            raise
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    results = raw["results"]
    assert [r["corpus_id"] for r in results] == [2, 0, 3, 1], f"Inspect {path}"
    assert all(a["score"] > b["score"] for a, b in pairwise(results)), f"Inspect {path}"
    assert (
        raw["detail"]["usage"]["requests"]
        == {"listwise": 1, "pointwise": 4, "pairwise": 6}[mode]
    )
    assert raw["detail"]["configuration"]["tokenizer"] == "google/embeddinggemma-300m"
