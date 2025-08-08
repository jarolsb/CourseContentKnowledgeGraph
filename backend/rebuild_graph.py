#!/usr/bin/env python3

import os
from knowledge_graph_builder import KnowledgeGraphBuilder

def rebuild_graph():
    print("\n" + "="*60)
    print("REBUILDING KNOWLEDGE GRAPH")
    print("="*60)
    
    builder = KnowledgeGraphBuilder()
    
    file_path = "chemistry_intro.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return
    
    print(f"\nProcessing {file_path}...")
    entities, relationships = builder.process_document(file_path)
    
    print(f"\nExtracted {len(entities)} entities and {len(relationships)} relationships")
    
    # Show sample relationships
    print("\nSample relationships extracted:")
    for rel in relationships[:5]:
        print(f"  {rel.source} --[{rel.type}]--> {rel.target}")
    
    print("\nCreating graph database...")
    builder.create_graph_database(entities, relationships)
    
    print("\nGenerating embeddings...")
    builder.create_vector_embeddings(entities)
    
    print("\nKnowledge graph rebuilt successfully!")
    
    # Verify relationships were created
    from neo4j import GraphDatabase
    from config import Config
    
    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        count = result.single()["count"]
        print(f"\nVerification: {count} relationships in database")
    
    driver.close()
    builder.close()

if __name__ == "__main__":
    rebuild_graph()