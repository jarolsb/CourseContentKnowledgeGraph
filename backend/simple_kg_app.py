#!/usr/bin/env python3

"""
Simplified Chemistry Knowledge Graph Application
This version provides a working demo with pre-extracted entities and relationships
"""

import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from config import Config
import numpy as np

class SimpleKnowledgeGraphApp:
    def __init__(self):
        Config.validate()
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        self.embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=Config.OPENAI_API_KEY
        )
    
    def setup_sample_graph(self):
        """Create a simplified knowledge graph with key chemistry concepts"""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create core chemistry entities
            entities = [
                ("Atom", "Concept", "The basic unit of matter"),
                ("Molecule", "Concept", "Two or more atoms bonded together"),
                ("Element", "Concept", "Pure substance made of one type of atom"),
                ("Compound", "Concept", "Substance made of two or more elements"),
                ("Chemical Bond", "Concept", "Force holding atoms together"),
                ("Ionic Bond", "BondType", "Transfer of electrons between atoms"),
                ("Covalent Bond", "BondType", "Sharing of electrons between atoms"),
                ("Metallic Bond", "BondType", "Sea of electrons around metal atoms"),
                ("Hydrogen", "Element", "Simplest element with 1 proton"),
                ("Oxygen", "Element", "Element with 8 protons"),
                ("Water", "Compound", "H2O - compound of hydrogen and oxygen"),
                ("Salt", "Compound", "NaCl - sodium chloride"),
                ("Chemical Reaction", "Process", "Transformation of substances"),
                ("pH", "Property", "Measure of acidity or basicity"),
                ("Acid", "Concept", "Substance that donates protons"),
                ("Base", "Concept", "Substance that accepts protons"),
            ]
            
            print("Creating entities...")
            for name, entity_type, description in entities:
                query = """
                CREATE (n:Entity {
                    name: $name,
                    type: $type,
                    description: $description
                })
                """
                session.run(query, {"name": name, "type": entity_type, "description": description})
            
            # Create relationships
            relationships = [
                ("Molecule", "COMPOSED_OF", "Atom"),
                ("Element", "CONSISTS_OF", "Atom"),
                ("Compound", "COMPOSED_OF", "Element"),
                ("Water", "COMPOSED_OF", "Hydrogen"),
                ("Water", "COMPOSED_OF", "Oxygen"),
                ("Salt", "FORMED_BY", "Ionic Bond"),
                ("Water", "FORMED_BY", "Covalent Bond"),
                ("Ionic Bond", "TYPE_OF", "Chemical Bond"),
                ("Covalent Bond", "TYPE_OF", "Chemical Bond"),
                ("Metallic Bond", "TYPE_OF", "Chemical Bond"),
                ("Chemical Reaction", "TRANSFORMS", "Compound"),
                ("Acid", "MEASURED_BY", "pH"),
                ("Base", "MEASURED_BY", "pH"),
                ("Hydrogen", "TYPE_OF", "Element"),
                ("Oxygen", "TYPE_OF", "Element"),
                ("Water", "TYPE_OF", "Compound"),
                ("Salt", "TYPE_OF", "Compound"),
            ]
            
            print("Creating relationships...")
            for source, rel_type, target in relationships:
                query = """
                MATCH (s:Entity {name: $source})
                MATCH (t:Entity {name: $target})
                CREATE (s)-[r:""" + rel_type + """]->(t)
                """
                try:
                    session.run(query, {"source": source, "target": target})
                except Exception as e:
                    print(f"Warning: Could not create relationship {source}->{target}: {e}")
            
            # Add embeddings
            print("Creating embeddings...")
            result = session.run("MATCH (n:Entity) RETURN n.name as name, n.description as description")
            for record in result:
                name = record["name"]
                description = record["description"]
                embedding = self.embeddings.embed_query(f"{name}: {description}")
                session.run(
                    "MATCH (n:Entity {name: $name}) SET n.embedding = $embedding",
                    {"name": name, "embedding": embedding}
                )
            
            print("Sample knowledge graph created successfully!")
    
    def semantic_search(self, query: str, k: int = 5):
        """Search for similar concepts using embeddings"""
        query_embedding = self.embeddings.embed_query(query)
        
        with self.driver.session() as session:
            # Get all entities with embeddings
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.embedding IS NOT NULL
                RETURN n.name as name, n.type as type, n.description as description, n.embedding as embedding
            """)
            
            # Calculate similarities
            similarities = []
            for record in result:
                entity_embedding = record["embedding"]
                similarity = np.dot(query_embedding, entity_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
                )
                similarities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "similarity": similarity
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:k]
    
    def get_entity_details(self, entity_name: str):
        """Get details and relationships for an entity"""
        with self.driver.session() as session:
            # Get entity details
            result = session.run("""
                MATCH (n:Entity {name: $name})
                RETURN n.name as name, n.type as type, n.description as description
            """, {"name": entity_name})
            
            entity = result.single()
            if not entity:
                return None
            
            # Get relationships
            rel_result = session.run("""
                MATCH (n:Entity {name: $name})-[r]-(connected:Entity)
                RETURN type(r) as rel_type, connected.name as connected_name, 
                       connected.type as connected_type,
                       CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END as direction
            """, {"name": entity_name})
            
            relationships = []
            for record in rel_result:
                relationships.append({
                    "type": record["rel_type"],
                    "connected": record["connected_name"],
                    "connected_type": record["connected_type"],
                    "direction": record["direction"]
                })
            
            return {
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "relationships": relationships
            }
    
    def answer_question(self, question: str):
        """Answer a question using the knowledge graph context"""
        # Search for relevant entities
        search_results = self.semantic_search(question, k=3)
        
        # Build context from search results
        context_parts = []
        for result in search_results:
            details = self.get_entity_details(result["name"])
            if details:
                context_parts.append(f"{details['name']} ({details['type']}): {details['description']}")
                for rel in details['relationships'][:3]:
                    context_parts.append(f"  - {rel['type']} {rel['connected']}")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a chemistry expert. Use the provided knowledge graph context 
            to answer the question accurately and concisely. If the context doesn't contain 
            enough information, say so."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return response.content
    
    def interactive_session(self):
        """Run an interactive query session"""
        print("\n" + "="*60)
        print("CHEMISTRY KNOWLEDGE GRAPH - INTERACTIVE SESSION")
        print("="*60)
        
        print("\nCommands:")
        print("  search <query>    - Search for similar concepts")
        print("  entity <name>     - Get entity details")
        print("  ask <question>    - Ask a natural language question")
        print("  list              - List all entities")
        print("  stats             - Show graph statistics")
        print("  exit              - Exit the session")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "exit":
                    print("Goodbye!")
                    break
                
                elif command == "search" and args:
                    results = self.semantic_search(args, k=3)
                    print(f"\nTop matches for '{args}':")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['name']} ({result['type']}): {result['description']}")
                
                elif command == "entity" and args:
                    details = self.get_entity_details(args)
                    if details:
                        print(f"\n{details['name']} ({details['type']})")
                        print(f"Description: {details['description']}")
                        if details['relationships']:
                            print("Relationships:")
                            for rel in details['relationships']:
                                arrow = "->" if rel['direction'] == 'outgoing' else "<-"
                                print(f"  {arrow} {rel['type']} {rel['connected']}")
                    else:
                        print(f"Entity '{args}' not found")
                
                elif command == "ask" and args:
                    answer = self.answer_question(args)
                    print(f"\nAnswer: {answer}")
                
                elif command == "list":
                    with self.driver.session() as session:
                        result = session.run("MATCH (n:Entity) RETURN n.name as name, n.type as type ORDER BY n.type, n.name")
                        current_type = None
                        for record in result:
                            if record["type"] != current_type:
                                current_type = record["type"]
                                print(f"\n{current_type}:")
                            print(f"  - {record['name']}")
                
                elif command == "stats":
                    with self.driver.session() as session:
                        nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                        rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                        print(f"\nGraph Statistics:")
                        print(f"  Nodes: {nodes}")
                        print(f"  Relationships: {rels}")
                
                else:
                    print("Invalid command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
    
    def close(self):
        self.driver.close()

def main():
    app = SimpleKnowledgeGraphApp()
    
    try:
        # Setup the sample graph
        print("Setting up sample chemistry knowledge graph...")
        app.setup_sample_graph()
        
        # Run interactive session
        app.interactive_session()
        
    finally:
        app.close()

if __name__ == "__main__":
    main()