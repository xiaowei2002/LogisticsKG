"""上传知识图谱到Neo4j数据库"""
import re
import os
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase, Driver
from src.models import Graph


def _sanitize_rel_type(predicate: str) -> str:
    """将谓词转换为合法的 Neo4j 关系类型。

    Neo4j 关系类型仅允许字母、数字和下划线，且不能以数字开头。
    中文字符保留，其余非法字符替换为下划线。
    """
    rel_type = re.sub(r"[^0-9A-Za-z_一-鿿]", "_", predicate)
    if not rel_type:
        rel_type = "REL"
    if rel_type[0].isdigit():
        rel_type = "REL_" + rel_type
    return rel_type


class Neo4jUploader:
    """负责将知识图谱上传到 Neo4j 数据库。"""
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        初始化 Neo4j 连接。

        Args:
            uri: Neo4j 连接 URI
            username: 数据库用户名
            password: 数据库密码
            database: 数据库名（默认 'neo4j'）
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None

    def connect(self) -> bool:
        """建立到 Neo4j 数据库的连接，返回是否成功。"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"成功连接到 Neo4j: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            return False

    def close(self):
        """关闭 Neo4j 连接。"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")

    def upload_graph(
        self,
        graph: Graph,
        graph_name: Optional[str] = None,
        clear_existing: bool = False,
        add_properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        上传知识图谱到 Neo4j。

        Args:
            graph: 要上传的 Graph 对象
            graph_name: 图谱的可选名称（作为属性存储）
            clear_existing: 是否先清除已有节点和关系
            add_properties: 添加到节点的额外属性

        Returns:
            bool: 上传是否成功
        """
        if not self.driver:
            logger.error("无活动的 Neo4j 连接，请先调用 connect()。")
            return False

        try:
            with self.driver.session(database=self.database) as session:
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("已清除现有图谱数据")

                node_count = self._create_nodes(
                    session, graph, graph_name, add_properties
                )
                rel_count = self._create_relationships(session, graph, graph_name)

                logger.info(
                    f"上传成功: {node_count} 个节点, {rel_count} 条关系"
                )
                return True

        except Exception as e:
            logger.error(f"上传图谱失败: {e}")
            return False

    def _create_nodes(
        self,
        session,
        graph: Graph,
        graph_name: Optional[str] = None,
        add_properties: Optional[Dict[str, Any]] = None,
    ) -> int:
        """从图谱实体创建 Neo4j 节点。"""
        properties = dict(add_properties or {})
        if graph_name:
            properties["graph_name"] = graph_name

        query = """
        UNWIND $entities AS entity
        MERGE (n:Entity {name: entity})
        SET n += $properties
        """

        result = session.run(
            query, entities=list(graph.entities), properties=properties
        )
        result.consume()
        return len(graph.entities)

    def _create_relationships(
        self, session, graph: Graph, graph_name: Optional[str] = None
    ) -> int:
        """从图谱关系创建 Neo4j 关系。"""
        rel_count = 0

        for subject, predicate, obj in graph.relations:
            rel_type = _sanitize_rel_type(predicate)

            query = """
            MATCH (s:Entity {name: $subject})
            MATCH (o:Entity {name: $object})
            MERGE (s)-[r:%s]->(o)
            SET r.predicate = $predicate
            """ % rel_type

            params = {
                "subject": subject,
                "object": obj,
                "predicate": predicate,
            }
            if graph_name:
                query += "\nSET r.graph_name = $graph_name"
                params["graph_name"] = graph_name

            session.run(query, **params)
            rel_count += 1

        return rel_count

    def query_graph(
        self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在 Neo4j 上执行 Cypher 查询，返回结果记录列表。"""
        if not self.driver:
            logger.error("无活动的 Neo4j 连接，请先调用 connect()。")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []

    def get_graph_stats(self) -> Dict[str, int]:
        """获取已上传图谱的基本统计信息。"""
        stats_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN
            count(DISTINCT n) AS node_count,
            count(DISTINCT r) AS relationship_count
        """

        results = self.query_graph(stats_query)
        if results:
            return {
                "node_count": results[0].get("node_count", 0),
                "relationship_count": results[0].get("relationship_count", 0),
            }
        return {"node_count": 0, "relationship_count": 0}


def upload_to_neo4j(
    graph: Graph,
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    graph_name: Optional[str] = None,
    clear_existing: bool = False,
    add_properties: Optional[Dict[str, Any]] = None,
) -> bool:
    uploader = Neo4jUploader(uri, username, password, database)

    try:
        if uploader.connect():
            return uploader.upload_graph(
                graph, graph_name, clear_existing, add_properties
            )
        return False
    finally:
        uploader.close()


def get_local_connection_config(
    host: str = "localhost",
    port: int = 7687,
    username: str = "neo4j",
    password: str = "password",
) -> Dict[str, str]:
    """获取本地 Neo4j 实例的连接配置。"""
    uri = f"bolt://{host}:{port}"
    return {"uri": uri, "username": username, "password": password, "database": "neo4j"}


if __name__ == "__main__":
    load_dotenv()

    graph = Graph(
        entities=["整车物流", "物流"],
        edges=["属于"],
        relations=[("整车物流", "属于", "物流")],
    )

    config = get_local_connection_config(
        host=os.getenv("NEO4J_HOST", "localhost"),
        port=int(os.getenv("NEO4J_PORT", "7687")),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    # 测试数据库连接
    uploader = Neo4jUploader(**config)
    try:
        if uploader.connect():
            logger.info("连接到Neo4j数据库...")
            # 上传图谱到数据库
            success = upload_to_neo4j(
                graph,
                uri=config["uri"],
                username=config["username"],
                password=config["password"],
                database=config["database"],
                graph_name="test",
                clear_existing=True,
            )
            print(f"上传结果: {success}")
        else:
            logger.info("连接失败...")

    finally:
        uploader.close()
        logger.info("断开数据库连接...")


