"""
将 pdfs/ 下的书籍与标准转换为知识图谱。
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src import pdf2json
from src.kg import KGGen
from src.models import Graph
from src.steps.deduplicate import DeduplicateMethod
from src.utils.json_cleand import clean_doc
from src.utils.neo4j_integration import upload_to_neo4j

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

CHUNK_SIZE = 16384
DEDUP_METHOD = DeduplicateMethod.SEMHASH
VISUALIZE = True


def json_to_graph(kg_gen: KGGen, json_path: Path, cache_dir: Path) -> Graph | None:
    """把单个结构化 JSON 展平后抽取为去重后的知识图谱。"""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    doc = clean_doc(doc)
    text = pdf2json.flatten_to_text(doc)
    if not text.strip():
        logger.warning("展平后为空，跳过: {}", json_path)
        return None

    logger.info("生成图谱: {}（{} 字符）", json_path.name, len(text))
    return kg_gen.generate(
        text,
        chunk_size=CHUNK_SIZE,
        deduplication_method=DEDUP_METHOD,
        context=json_path.stem,
        cache_dir=str(cache_dir),
        no_dspy=True,
    )


def _load_or_generate(
    kg_gen: KGGen, json_path: Path, graph_json: Path, cache_dir: Path, force: bool
) -> Graph | None:
    """优先读取缓存（_graph.json），命中则跳过"""
    if graph_json.exists() and not force:
        try:
            graph = Graph.from_file(str(graph_json))
            logger.info("命中缓存，跳过生成: {}", graph_json.name)
            return graph
        except Exception:
            logger.exception("缓存读取失败，回退重新生成: {}", graph_json.name)

    graph = json_to_graph(kg_gen, json_path, cache_dir)
    if graph is not None:
        kg_gen.export_graph(graph, str(graph_json))
    return graph


def upload_merged_to_neo4j(graph: Graph, clear_existing: bool = False) -> bool:
    """把最终合并后的图谱上传到 Neo4j，连接参数从环境变量读取。"""
    uri = f"bolt://{os.getenv('NEO4J_HOST', 'localhost')}:{os.getenv('NEO4J_PORT', '7687')}"
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    return upload_to_neo4j(
        graph,
        uri=uri,
        username=username,
        password=password,
        database=database,
        graph_name="merged",
        clear_existing=clear_existing,
    )


def build_graphs_from_json(output_dir: Path, force: bool = False) -> list[Graph]:
    """遍历 output 下的结构化 JSON，逐个生成（或读缓存）图谱。"""
    kg_gen = KGGen()
    cache_dir = output_dir / ".kg_cache"
    json_files = [
        p for p in sorted(output_dir.glob("*.json")) if not p.stem.endswith("_graph")
    ]
    logger.info("共发现 {} 个结构化 JSON", len(json_files))

    graphs: list[Graph] = []
    for json_path in json_files:
        stem = json_path.stem
        graph_json = output_dir / f"{stem}_graph.json"
        try:
            graph = _load_or_generate(kg_gen, json_path, graph_json, cache_dir, force)
        except Exception:
            logger.exception("处理失败，跳过: {}", json_path)
            continue
        if graph is None:
            continue

        graphs.append(graph)
        if VISUALIZE:
            kg_gen.visualize(graph, str(output_dir / f"{stem}_graph.html"))
        logger.info(
            "已保存 {}：{} 实体 / {} 关系",
            stem,
            len(graph.entities),
            len(graph.relations),
        )
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → 结构化 JSON → 知识图谱 流水线")
    parser.add_argument("--input", type=str, default="pdfs", help="PDF 目录（默认 pdfs）")
    parser.add_argument("--output", type=str, default="output", help="输出目录（默认 output）")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 渲染分辨率（默认 150）")
    parser.add_argument("--vl-model", type=str, default=None, help="视觉模型名，缺省读取 OPENAI_VL_MODEL_NAME")
    parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF→JSON，直接用已有 JSON 建图谱")
    parser.add_argument("--force", action="store_true", help="忽略缓存，重新生成图谱")
    parser.add_argument("--upload-neo4j", action="store_true", help="生成完成后上传合并图谱到 Neo4j")
    parser.add_argument("--neo4j-clear", action="store_true", help="上传前清空 Neo4j 现有数据")
    args = parser.parse_args()

    load_dotenv()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: PDF → 结构化 JSON
    if not args.skip_pdf:
        pdf2json.process_folder(
            args.input, str(output_dir), args.dpi, args.vl_model
        )

    # Stage 2: 结构化 JSON → 知识图谱
    graphs = build_graphs_from_json(output_dir, force=args.force)

    # 合并所有文档图谱为一个总图谱
    if graphs:
        kg_gen = KGGen()
        merged = kg_gen.aggregate(graphs)
        kg_gen.export_graph(merged, str(output_dir / "merged_graph.json"))
        if VISUALIZE:
            kg_gen.visualize(merged, str(output_dir / "merged_graph.html"))
        logger.info(
            "合并图谱：{} 实体 / {} 关系",
            len(merged.entities),
            len(merged.relations),
        )

        if args.upload_neo4j:
            if upload_merged_to_neo4j(merged, clear_existing=args.neo4j_clear):
                logger.info("合并图谱已上传到 Neo4j")
            else:
                logger.error("合并图谱上传到 Neo4j 失败")

    logger.info("完成，共生成 {} 个文档图谱", len(graphs))


if __name__ == "__main__":
    main()
