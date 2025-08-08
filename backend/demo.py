#!/usr/bin/env python3

"""
Demo script showcasing the Chemistry Knowledge Graph capabilities
"""

from simple_kg_app import SimpleKnowledgeGraphApp
import time

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def demo():
    app = SimpleKnowledgeGraphApp()
    
    try:
        # Setup
        print_section("SETTING UP CHEMISTRY KNOWLEDGE GRAPH")
        app.setup_sample_graph()
        time.sleep(1)
        
        # Show statistics
        print_section("GRAPH STATISTICS")
        with app.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"Created {nodes} entities and {rels} relationships")
        
        # Demo 1: Semantic Search
        print_section("DEMO 1: SEMANTIC SEARCH")
        query = "chemical bonds"
        print(f"Searching for: '{query}'")
        results = app.semantic_search(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']} - {result['description'][:50]}...")
        
        # Demo 2: Entity Details
        print_section("DEMO 2: ENTITY EXPLORATION")
        entity = "Water"
        print(f"Exploring entity: '{entity}'")
        details = app.get_entity_details(entity)
        if details:
            print(f"\nName: {details['name']}")
            print(f"Type: {details['type']}")
            print(f"Description: {details['description']}")
            print("\nRelationships:")
            for rel in details['relationships']:
                arrow = "->" if rel['direction'] == 'outgoing' else "<-"
                print(f"  {arrow} {rel['type']} {rel['connected']}")
        
        # Demo 3: Natural Language Q&A
        print_section("DEMO 3: NATURAL LANGUAGE Q&A")
        questions = [
            "What is the difference between ionic and covalent bonds?",
            "How is water formed?",
            "What are acids and bases?"
        ]
        
        for q in questions:
            print(f"\nQ: {q}")
            answer = app.answer_question(q)
            print(f"A: {answer}\n")
            time.sleep(1)
        
        # Demo 4: Graph Traversal
        print_section("DEMO 4: GRAPH TRAVERSAL")
        with app.driver.session() as session:
            # Find path from Atom to Water
            result = session.run("""
                MATCH path = (a:Entity {name: 'Atom'})-[*..3]-(w:Entity {name: 'Water'})
                RETURN [node in nodes(path) | node.name] as nodes
                LIMIT 1
            """)
            record = result.single()
            if record:
                path = record["nodes"]
                print(f"Path from Atom to Water: {' -> '.join(path)}")
        
        # Demo 5: Complex Query
        print_section("DEMO 5: COMPLEX QUERY - BOND TYPES")
        with app.driver.session() as session:
            result = session.run("""
                MATCH (bond:Entity)-[:TYPE_OF]->(cb:Entity {name: 'Chemical Bond'})
                MATCH (compound:Entity)-[:FORMED_BY]->(bond)
                RETURN bond.name as bond_type, collect(compound.name) as compounds
            """)
            print("\nBond types and their compounds:")
            for record in result:
                print(f"  {record['bond_type']}: {', '.join(record['compounds'])}")
        
        print_section("DEMO COMPLETE!")
        print("\nThis knowledge graph can be used to:")
        print("1. Provide context for Claude or other LLMs")
        print("2. Answer chemistry questions with structured knowledge")
        print("3. Explore relationships between chemical concepts")
        print("4. Build educational applications")
        print("\nRun 'python simple_kg_app.py' for interactive mode!")
        
    finally:
        app.close()

if __name__ == "__main__":
    demo()