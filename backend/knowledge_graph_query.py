from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from neo4j import GraphDatabase
import numpy as np
from config import Config

class KnowledgeGraphQuery:
    def __init__(self):
        Config.validate()
        self.embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=Config.NEO4J_URI,
            username=Config.NEO4J_USERNAME,
            password=Config.NEO4J_PASSWORD,
            index_name="entity_index",
            node_label="Entity",
            text_node_properties=["name", "type"],
            embedding_node_property="embedding"
        )
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def cypher_query(self, query: str) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    
    def get_entity_relationships(self, entity_name: str) -> Dict[str, Any]:
        query = """
        MATCH (e {name: $name})-[r]-(connected)
        RETURN e.name as entity, 
               type(r) as relationship_type, 
               connected.name as connected_entity,
               connected.type as connected_type,
               properties(r) as relationship_properties
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"name": entity_name})
            relationships = [record.data() for record in result]
            
        return {
            "entity": entity_name,
            "relationships": relationships
        }
    
    def find_path(self, start_entity: str, end_entity: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        query = """
        MATCH path = shortestPath((start {name: $start})-[*..%d]-(end {name: $end}))
        RETURN [node in nodes(path) | node.name] as nodes,
               [rel in relationships(path) | type(rel)] as relationships
        """ % max_depth
        
        with self.driver.session() as session:
            result = session.run(query, {"start": start_entity, "end": end_entity})
            paths = [record.data() for record in result]
            
        return paths
    
    def get_entity_details(self, entity_name: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (e {name: $name})
        RETURN e as entity, labels(e) as labels
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"name": entity_name})
            record = result.single()
            
            if record:
                entity_data = dict(record["entity"])
                entity_data["labels"] = record["labels"]
                return entity_data
            return None
    
    def search_by_type(self, entity_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (e:{entity_type.replace(' ', '_')})
        RETURN e.name as name, e.type as type, properties(e) as properties
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"limit": limit})
            return [record.data() for record in result]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_types": "MATCH (n) RETURN labels(n) as labels, count(n) as count",
            "relationship_types": "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        }
        
        stats = {}
        with self.driver.session() as session:
            for key, query in queries.items():
                result = session.run(query)
                if key in ["total_nodes", "total_relationships"]:
                    stats[key] = result.single()["count"]
                else:
                    stats[key] = [record.data() for record in result]
        
        return stats
    
    def close(self):
        self.driver.close()