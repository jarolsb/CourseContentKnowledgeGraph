from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
import json
from config import Config

class Entity(BaseModel):
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type of entity (e.g., Concept, Process, Element, Principle)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")

class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")

class KnowledgeGraphExtraction(BaseModel):
    entities: List[Entity] = Field(description="List of entities extracted")
    relationships: List[Relationship] = Field(description="List of relationships extracted")

class KnowledgeGraphBuilder:
    def __init__(self):
        Config.validate()
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=Config.OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_knowledge_graph(self, text: str) -> KnowledgeGraphExtraction:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting knowledge graphs from educational text.
            Extract entities and relationships from the chemistry text.
            
            For entities, identify:
            - Chemical concepts (atoms, molecules, bonds, etc.)
            - Chemical processes (reactions, phase changes, etc.)
            - Chemical elements and compounds
            - Scientific principles and laws
            - Important scientists or discoveries
            
            For relationships, identify connections like:
            - "consists_of", "contains", "composed_of"
            - "transforms_to", "reacts_with", "produces"
            - "has_property", "exhibits", "characterized_by"
            - "discovered_by", "defined_by", "explained_by"
            - "example_of", "type_of", "category_of"
            - "causes", "enables", "prevents"
            
            Return the results in this JSON format:
            {{
                "entities": [
                    {{"name": "...", "type": "...", "properties": {{...}}}},
                    ...
                ],
                "relationships": [
                    {{"source": "...", "target": "...", "type": "...", "properties": {{...}}}},
                    ...
                ]
            }}
            """),
            ("human", "Extract entities and relationships from this text:\n\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text})
        
        try:
            data = json.loads(result)
            return KnowledgeGraphExtraction(**data)
        except Exception as e:
            print(f"Error parsing extraction result: {e}")
            return KnowledgeGraphExtraction(entities=[], relationships=[])
    
    def process_document(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        chunks = self.text_splitter.split_text(content)
        
        all_entities = {}
        all_relationships = []
        
        print(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            extraction = self.extract_knowledge_graph(chunk)
            
            for entity in extraction.entities:
                if entity.name not in all_entities:
                    all_entities[entity.name] = entity
            
            for rel in extraction.relationships:
                all_relationships.append(rel)
        
        return list(all_entities.values()), all_relationships
    
    def create_graph_database(self, entities: List[Entity], relationships: List[Relationship]):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            print("Creating entities in Neo4j...")
            for entity in entities:
                properties = entity.properties or {}
                # Escape property names with backticks and sanitize keys
                props_list = []
                params = {"name": entity.name}
                for k, v in properties.items():
                    # Replace spaces and special characters in parameter names
                    safe_key = k.replace(' ', '_').replace('-', '_')
                    props_list.append(f"`{k}`: ${safe_key}")
                    params[safe_key] = v
                
                props_str = ", ".join(props_list)
                if props_str:
                    props_str = ", " + props_str
                
                # Sanitize entity type for use as label
                label = entity.type.replace(' ', '_').replace('-', '_')
                
                query = f"""
                CREATE (e:{label} {{
                    name: $name{props_str}
                }})
                """
                try:
                    session.run(query, params)
                except Exception as e:
                    print(f"Error creating entity {entity.name}: {e}")
                    # Try without properties if there's an error
                    query = f"""
                    CREATE (e:{label} {{
                        name: $name
                    }})
                    """
                    session.run(query, {"name": entity.name})
            
            print("Creating relationships in Neo4j...")
            for rel in relationships:
                properties = rel.properties or {}
                # Escape property names with backticks and sanitize keys
                props_list = []
                params = {"source": rel.source, "target": rel.target}
                for k, v in properties.items():
                    # Replace spaces and special characters in parameter names
                    safe_key = k.replace(' ', '_').replace('-', '_')
                    props_list.append(f"`{k}`: ${safe_key}")
                    params[safe_key] = v
                
                props_str = ", ".join(props_list)
                if props_str:
                    props_str = "{" + props_str + "}"
                else:
                    props_str = ""
                
                # Sanitize relationship type
                rel_type = rel.type.upper().replace(' ', '_').replace('-', '_')
                
                query = f"""
                MATCH (source {{name: $source}})
                MATCH (target {{name: $target}})
                CREATE (source)-[r:{rel_type} {props_str}]->(target)
                """
                try:
                    session.run(query, params)
                except Exception as e:
                    print(f"Error creating relationship {rel.source} -> {rel.target}: {e}")
    
    def create_vector_embeddings(self, entities: List[Entity]):
        with self.driver.session() as session:
            print("Creating vector embeddings...")
            for entity in entities:
                description = f"{entity.name} ({entity.type})"
                if entity.properties:
                    description += f" - {json.dumps(entity.properties)}"
                
                embedding = self.embeddings.embed_query(description)
                
                query = """
                MATCH (e {name: $name})
                SET e.embedding = $embedding
                """
                session.run(query, {"name": entity.name, "embedding": embedding})
    
    def close(self):
        self.driver.close()