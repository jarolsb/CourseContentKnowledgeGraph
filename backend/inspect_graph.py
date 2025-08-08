#!/usr/bin/env python3

from neo4j import GraphDatabase
from config import Config

def inspect_graph():
    Config.validate()
    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Get sample nodes
        print("\nSAMPLE NODES:")
        print("-" * 40)
        result = session.run("MATCH (n) RETURN n LIMIT 10")
        for record in result:
            node = record["n"]
            labels = list(node.labels)
            props = dict(node)
            print(f"Labels: {labels}, Properties: {props}")
        
        # Get all unique labels
        print("\nALL NODE LABELS:")
        print("-" * 40)
        result = session.run("MATCH (n) RETURN DISTINCT labels(n) as labels")
        for record in result:
            print(f"  - {record['labels']}")
        
        # Get sample relationships
        print("\nSAMPLE RELATIONSHIPS:")
        print("-" * 40)
        result = session.run("MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as type, b.name as target LIMIT 10")
        count = 0
        for record in result:
            print(f"  {record['source']} --[{record['type']}]--> {record['target']}")
            count += 1
        if count == 0:
            print("  No relationships found")
        
        # Search for specific entities
        print("\nSEARCHING FOR KEY ENTITIES:")
        print("-" * 40)
        entities_to_search = ["atom", "molecule", "hydrogen", "water", "bond"]
        for entity_name in entities_to_search:
            result = session.run("MATCH (n) WHERE n.name =~ $pattern RETURN n.name as name, labels(n) as labels LIMIT 1", 
                                {"pattern": f"(?i).*{entity_name}.*"})
            record = result.single()
            if record:
                print(f"  Found '{entity_name}': {record['name']} (Labels: {record['labels']})")
            else:
                print(f"  Not found: '{entity_name}'")
    
    driver.close()

if __name__ == "__main__":
    inspect_graph()