#!/usr/bin/env python3

from knowledge_graph_query import KnowledgeGraphQuery
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import Config

def test_knowledge_graph_queries():
    print("\n" + "="*60)
    print("TESTING KNOWLEDGE GRAPH QUERIES")
    print("="*60)
    
    # Initialize query engine
    query_engine = KnowledgeGraphQuery()
    
    # Initialize LLM for natural language queries
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Test 1: Graph Statistics
    print("\n1. GRAPH STATISTICS:")
    print("-" * 40)
    stats = query_engine.get_graph_statistics()
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    
    # Test 2: Semantic Search
    print("\n2. SEMANTIC SEARCH - 'chemical bonds':")
    print("-" * 40)
    results = query_engine.similarity_search("chemical bonds", k=3)
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")
    
    # Test 3: Entity Details
    print("\n3. ENTITY DETAILS - 'hydrogen':")
    print("-" * 40)
    entity = query_engine.get_entity_details("hydrogen")
    if entity:
        print(f"Found: {entity.get('name', 'Unknown')}")
        print(f"Labels: {entity.get('labels', [])}")
    else:
        print("Entity 'hydrogen' not found")
    
    # Test 4: Find Path
    print("\n4. FINDING PATH - 'atom' to 'molecule':")
    print("-" * 40)
    paths = query_engine.find_path("atom", "molecule")
    if paths:
        for path in paths[:1]:  # Show first path only
            if 'nodes' in path:
                print(f"Path: {' -> '.join(path['nodes'])}")
    else:
        print("No path found")
    
    # Test 5: Natural Language Question
    print("\n5. NATURAL LANGUAGE QUESTION:")
    print("-" * 40)
    question = "What is the difference between ionic and covalent bonds?"
    print(f"Question: {question}")
    
    # Get context from knowledge graph
    search_results = query_engine.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in search_results])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a chemistry expert assistant. Use the provided context from the knowledge graph 
        to answer the user's question. Be concise and accurate."""),
        ("human", "Context from knowledge graph:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    print(f"Answer: {response.content}")
    
    # Test 6: Search by Type
    print("\n6. SEARCH BY TYPE - 'Concept':")
    print("-" * 40)
    entities = query_engine.search_by_type("Concept", limit=5)
    if entities:
        print("Concepts found:")
        for entity in entities:
            print(f"  - {entity.get('name', 'Unknown')}")
    else:
        print("No entities of type 'Concept' found")
    
    # Test 7: Entity Relationships
    print("\n7. ENTITY RELATIONSHIPS - 'water':")
    print("-" * 40)
    relationships = query_engine.get_entity_relationships("water")
    if relationships['relationships']:
        print(f"Relationships for 'water':")
        for rel in relationships['relationships'][:5]:  # Show first 5
            print(f"  - {rel['relationship_type']} -> {rel['connected_entity']}")
    else:
        print("No relationships found for 'water'")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    
    # Close connection
    query_engine.close()

if __name__ == "__main__":
    try:
        Config.validate()
        test_knowledge_graph_queries()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()