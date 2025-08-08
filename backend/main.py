#!/usr/bin/env python3

import os
import sys
import json
from typing import Optional
from knowledge_graph_builder import KnowledgeGraphBuilder
from knowledge_graph_query import KnowledgeGraphQuery
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import Config

class ChemistryKnowledgeGraphApp:
    def __init__(self):
        try:
            Config.validate()
            self.builder = KnowledgeGraphBuilder()
            self.query_engine = KnowledgeGraphQuery()
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                api_key=Config.OPENAI_API_KEY
            )
        except ValueError as e:
            print(f"Configuration error: {e}")
            print("Please ensure you have created a .env file with the required API keys.")
            print("Copy .env.example to .env and fill in your credentials.")
            sys.exit(1)
    
    def build_graph(self, rebuild: bool = False):
        print("\n" + "="*60)
        print("BUILDING KNOWLEDGE GRAPH FROM CHEMISTRY TEXT")
        print("="*60)
        
        if not rebuild:
            stats = self.query_engine.get_graph_statistics()
            if stats['total_nodes'] > 0:
                print(f"\nGraph already exists with {stats['total_nodes']} nodes.")
                response = input("Do you want to rebuild it? (y/n): ").lower()
                if response != 'y':
                    print("Using existing graph.")
                    return
        
        file_path = "chemistry_intro.txt"
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return
        
        print(f"\nProcessing {file_path}...")
        entities, relationships = self.builder.process_document(file_path)
        
        print(f"\nExtracted {len(entities)} entities and {len(relationships)} relationships")
        
        print("\nCreating graph database...")
        self.builder.create_graph_database(entities, relationships)
        
        print("\nGenerating embeddings...")
        self.builder.create_vector_embeddings(entities)
        
        print("\nKnowledge graph built successfully!")
    
    def interactive_query(self):
        print("\n" + "="*60)
        print("CHEMISTRY KNOWLEDGE GRAPH QUERY INTERFACE")
        print("="*60)
        print("\nAvailable commands:")
        print("  1. search <query>       - Semantic search for concepts")
        print("  2. entity <name>        - Get entity details and relationships")
        print("  3. path <from> <to>     - Find path between two entities")
        print("  4. type <type>          - List entities of a specific type")
        print("  5. stats                - Show graph statistics")
        print("  6. ask <question>       - Ask a natural language question")
        print("  7. cypher <query>       - Execute custom Cypher query")
        print("  8. rebuild              - Rebuild the knowledge graph")
        print("  9. help                 - Show this help message")
        print("  10. exit                - Exit the application")
        
        while True:
            try:
                print("\n" + "-"*40)
                user_input = input("\nEnter command: ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "exit":
                    print("Goodbye!")
                    break
                
                elif command == "help":
                    self.interactive_query()
                    break
                
                elif command == "search":
                    if not args:
                        print("Please provide a search query.")
                        continue
                    
                    print(f"\nSearching for: {args}")
                    results = self.query_engine.similarity_search(args, k=5)
                    
                    if results:
                        print(f"\nFound {len(results)} similar concepts:")
                        for i, doc in enumerate(results, 1):
                            print(f"  {i}. {doc.page_content}")
                    else:
                        print("No results found.")
                
                elif command == "entity":
                    if not args:
                        print("Please provide an entity name.")
                        continue
                    
                    details = self.query_engine.get_entity_details(args)
                    if details:
                        print(f"\nEntity: {args}")
                        print(f"Type: {details.get('type', 'Unknown')}")
                        if 'properties' in details:
                            print("Properties:", json.dumps(details['properties'], indent=2))
                        
                        relationships = self.query_engine.get_entity_relationships(args)
                        if relationships['relationships']:
                            print(f"\nRelationships:")
                            for rel in relationships['relationships']:
                                print(f"  - {rel['relationship_type']} -> {rel['connected_entity']} ({rel['connected_type']})")
                    else:
                        print(f"Entity '{args}' not found.")
                
                elif command == "path":
                    entities = args.split()
                    if len(entities) != 2:
                        print("Please provide exactly two entity names.")
                        continue
                    
                    paths = self.query_engine.find_path(entities[0], entities[1])
                    if paths:
                        print(f"\nPath from '{entities[0]}' to '{entities[1]}':")
                        for path in paths:
                            nodes = path['nodes']
                            rels = path['relationships']
                            path_str = nodes[0]
                            for i, rel in enumerate(rels):
                                path_str += f" --[{rel}]-> {nodes[i+1]}"
                            print(f"  {path_str}")
                    else:
                        print(f"No path found between '{entities[0]}' and '{entities[1]}'.")
                
                elif command == "type":
                    if not args:
                        print("Please provide an entity type.")
                        continue
                    
                    entities = self.query_engine.search_by_type(args, limit=20)
                    if entities:
                        print(f"\nEntities of type '{args}':")
                        for entity in entities:
                            print(f"  - {entity['name']}")
                    else:
                        print(f"No entities of type '{args}' found.")
                
                elif command == "stats":
                    stats = self.query_engine.get_graph_statistics()
                    print("\nGraph Statistics:")
                    print(f"  Total Nodes: {stats['total_nodes']}")
                    print(f"  Total Relationships: {stats['total_relationships']}")
                    
                    if stats['node_types']:
                        print("\n  Node Types:")
                        for item in stats['node_types']:
                            labels = item['labels'][0] if item['labels'] else 'Unknown'
                            print(f"    - {labels}: {item['count']}")
                    
                    if stats['relationship_types']:
                        print("\n  Relationship Types:")
                        for item in stats['relationship_types']:
                            print(f"    - {item['type']}: {item['count']}")
                
                elif command == "ask":
                    if not args:
                        print("Please provide a question.")
                        continue
                    
                    print(f"\nQuestion: {args}")
                    
                    search_results = self.query_engine.similarity_search(args, k=3)
                    context = "\n".join([doc.page_content for doc in search_results])
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a chemistry expert assistant. Use the provided context from the knowledge graph 
                        to answer the user's question. If the context doesn't contain enough information, 
                        say so and provide what information you can based on the context."""),
                        ("human", "Context from knowledge graph:\n{context}\n\nQuestion: {question}")
                    ])
                    
                    chain = prompt | self.llm
                    response = chain.invoke({"context": context, "question": args})
                    
                    print(f"\nAnswer: {response.content}")
                
                elif command == "cypher":
                    if not args:
                        print("Please provide a Cypher query.")
                        continue
                    
                    try:
                        results = self.query_engine.cypher_query(args)
                        if results:
                            print(f"\nQuery results ({len(results)} records):")
                            for record in results[:10]:
                                print(f"  {json.dumps(record, indent=2)}")
                            if len(results) > 10:
                                print(f"  ... and {len(results) - 10} more records")
                        else:
                            print("Query returned no results.")
                    except Exception as e:
                        print(f"Error executing query: {e}")
                
                elif command == "rebuild":
                    self.build_graph(rebuild=True)
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit the application.")
            except Exception as e:
                print(f"Error: {e}")
    
    def close(self):
        self.builder.close()
        self.query_engine.close()

def main():
    print("\n" + "="*60)
    print("CHEMISTRY KNOWLEDGE GRAPH APPLICATION")
    print("="*60)
    
    app = ChemistryKnowledgeGraphApp()
    
    try:
        app.build_graph()
        app.interactive_query()
    finally:
        app.close()

if __name__ == "__main__":
    main()