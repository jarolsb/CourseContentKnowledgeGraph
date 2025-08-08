# Chemistry Knowledge Graph Application

A Python application that extracts knowledge from chemistry text documents and builds a queryable Neo4j knowledge graph with vector embeddings for semantic search.

## Features

- Automatic knowledge graph extraction from text documents
- Entity and relationship identification using GPT-4
- Neo4j graph database storage
- Vector embeddings for semantic similarity search
- Interactive console interface for querying
- Natural language question answering using graph context

## Prerequisites

1. **Python 3.10+**
2. **Neo4j Database** (Community or Enterprise Edition)
   - Download from: https://neo4j.com/download/
   - Default connection: bolt://localhost:7687
3. **OpenAI API Key**
   - Get from: https://platform.openai.com/api-keys

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

3. Start Neo4j database:
   - Open Neo4j Desktop
   - Create a new project and database
   - Start the database
   - Note the password you set

## Usage

Run the application:
```bash
python main.py
```

### Available Commands

1. **search <query>** - Semantic search for chemistry concepts
   ```
   search atomic bonds
   ```

2. **entity <name>** - Get details about a specific entity
   ```
   entity hydrogen
   ```

3. **path <from> <to>** - Find connections between entities
   ```
   path atom molecule
   ```

4. **type <type>** - List all entities of a specific type
   ```
   type Element
   ```

5. **stats** - Show graph statistics

6. **ask <question>** - Ask natural language questions
   ```
   ask What is the difference between ionic and covalent bonds?
   ```

7. **cypher <query>** - Execute custom Cypher queries
   ```
   cypher MATCH (n:Element) RETURN n.name LIMIT 5
   ```

8. **rebuild** - Rebuild the knowledge graph from source

9. **help** - Show available commands

10. **exit** - Exit the application

## Project Structure

- `main.py` - Main application with console interface
- `knowledge_graph_builder.py` - Extracts entities and relationships from text
- `knowledge_graph_query.py` - Query interface for the graph database
- `config.py` - Configuration management
- `chemistry_intro.txt` - Source chemistry text document
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## How It Works

1. **Text Processing**: The chemistry text is split into chunks for processing
2. **Entity Extraction**: GPT-4 identifies chemical concepts, processes, and relationships
3. **Graph Creation**: Entities and relationships are stored in Neo4j
4. **Embeddings**: Vector embeddings are generated for semantic search
5. **Querying**: Users can query using natural language or graph traversal

## Example Queries

- Find all chemical elements mentioned
- Trace the relationship between atoms and molecules
- Search for information about chemical bonds
- Ask questions about chemical processes
- Explore phase transitions and states of matter

## Troubleshooting

1. **Neo4j Connection Error**: 
   - Ensure Neo4j is running
   - Check credentials in `.env`
   - Verify URI format (bolt://localhost:7687)

2. **OpenAI API Error**:
   - Verify API key is valid
   - Check API quota/limits

3. **Import Errors**:
   - Ensure all dependencies are installed
   - Use Python 3.10 or higher

## Future Enhancements

- Web interface for visualization
- Support for multiple document formats
- Advanced graph algorithms
- Integration with Claude API for enhanced responses
- Export functionality for graph data