# HTMA CLI & Demo

This document describes the HTMA command-line interface and interactive demo.

## Installation

First, ensure all dependencies are installed:

```bash
pip install -e ".[dev]"
```

## Prerequisites

The HTMA CLI requires a running Ollama instance for LLM operations.

1. Install Ollama from https://ollama.ai
2. Start Ollama: `ollama serve`
3. Pull required models:
   ```bash
   ollama pull llama3:8b
   ollama pull mistral:7b
   ollama pull nomic-embed-text
   ```

## CLI Commands

Once installed, you can use the `htma` command:

### Interactive Chat

Start a memory-augmented conversation:

```bash
htma chat
```

Options:
- `--ollama-url`: Ollama API URL (default: http://localhost:11434)
- `--reasoner`: Model for reasoning/LLM₁ (default: llama3:8b)
- `--curator`: Model for memory curation/LLM₂ (default: mistral:7b)

Example:
```bash
htma chat --reasoner llama3:8b --curator mistral:7b
```

### Query Memory

Search memory without starting a conversation:

```bash
htma query "What are my favorite programming languages?"
```

Options:
- `--limit, -n`: Maximum results (default: 10)
- `--semantic/--no-semantic`: Include semantic memory (default: true)
- `--episodic/--no-episodic`: Include episodic memory (default: true)

Examples:
```bash
# Query with custom limit
htma query "my hobbies" --limit 5

# Query only semantic memory
htma query "my job" --no-episodic

# Query only episodic memory
htma query "what did we discuss yesterday" --no-semantic
```

### Consolidation

Trigger memory consolidation to create abstractions and detect patterns:

```bash
htma consolidate
```

Options:
- `--force, -f`: Force consolidation even if not needed

The consolidation process:
- Creates higher-level summaries from raw episodes
- Detects recurring patterns
- Strengthens frequently co-accessed links
- Prunes weak links

### Memory Status

View current memory statistics:

```bash
htma status
```

Shows:
- Working memory utilization
- Number of entities and facts (semantic memory)
- Episode counts by level (episodic memory)
- Last consolidation time

### Export Memory

Export all memory to a JSON file:

```bash
htma export
```

Options:
- `--output, -o`: Output file path (default: htma_export_TIMESTAMP.json)

Example:
```bash
htma export --output my_memory.json
```

The export includes:
- All entities and facts (semantic memory)
- All episodes and links (episodic memory)
- Statistics and metadata

### Reset Memory

**DANGEROUS:** Delete all memory:

```bash
htma reset
```

Options:
- `--force, -f`: Skip confirmation prompt

This command:
- Deletes all entities and facts
- Deletes all episodes and links
- Clears vector embeddings
- Cannot be undone!

## Guided Demo

The guided demo script provides an interactive walkthrough of HTMA capabilities:

```bash
python scripts/demo.py
```

### Demo Scenarios

1. **Introduction & Basic Conversation**
   - Basic interaction with memory-augmented agent
   - Information storage

2. **Teaching Facts & Memory Retrieval**
   - Teaching the agent facts
   - Retrieving facts in later questions
   - Demonstration of semantic memory

3. **Temporal Reasoning**
   - Handling facts that change over time
   - Bi-temporal validity tracking
   - Historical queries

4. **Pattern Detection**
   - Identifying recurring behaviors
   - Pattern emergence from repeated mentions

5. **Memory Consolidation**
   - Creating abstractions from raw memories
   - Pattern detection across episodes
   - Link maintenance

### Demo Features

- Interactive scenario selection
- Guided walkthrough of each feature
- Free chat mode for experimentation
- Separate demo database (doesn't affect main HTMA data)

The demo data is stored in `~/.htma_demo/` and can be safely deleted after the demo.

## Data Storage

### Default Locations

- Main CLI data: `~/.htma/`
  - SQLite database: `~/.htma/htma.db`
  - ChromaDB vectors: `~/.htma/chroma/`

- Demo data: `~/.htma_demo/`
  - SQLite database: `~/.htma_demo/htma.db`
  - ChromaDB vectors: `~/.htma_demo/chroma/`

### Backup Your Data

Before using `htma reset`, you can backup your memory:

```bash
# Export to JSON
htma export --output backup.json

# Or copy the data directory
cp -r ~/.htma ~/.htma_backup
```

## Troubleshooting

### "Cannot connect to Ollama"

Ensure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve
```

### "Model not found"

Pull the required models:
```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull nomic-embed-text
```

### Slow performance

- Use smaller models (e.g., `llama3:8b` instead of `llama3:70b`)
- Reduce retrieval limits in queries
- Run consolidation less frequently

### Memory growing too large

- Use `htma consolidate` to create abstractions and prune old memories
- Use `htma export` to backup and `htma reset` to start fresh
- Adjust `consolidation_strength` values in episodes

## Advanced Usage

### Custom Models

You can use different models for different components:

```bash
# Use larger model for reasoning, smaller for curation
htma chat --reasoner llama3:70b --curator mistral:7b

# Use a different embedding model (requires reindexing)
# Edit DEFAULT_EMBEDDING_MODEL in cli.py
```

### Programmatic Usage

The CLI can also be used as a library:

```python
from htma.cli import initialize_system

# Initialize HTMA
agent, memory, consolidation, llm = await initialize_system()

# Use the agent
response = await agent.process_message("Hello!")
print(response.message)

# Query memory
result = await memory.query("my interests")

# Run consolidation
report = await consolidation.run_cycle()
```

## Examples

### Building a Personal Memory

```bash
# Start chatting
htma chat

# Tell the agent about yourself
> My name is Alex and I'm a software engineer
> I work on ML projects in Python
> I have two cats named Whiskers and Luna
> I love hiking on weekends

# Exit and query later
> exit

# Query what the agent remembers
htma query "what do you know about me?"

# Check memory stats
htma status

# Run consolidation
htma consolidate
```

### Memory Evolution Over Time

```bash
# Day 1: Initial fact
htma chat
> I work at TechCorp as a Senior Engineer
> exit

# Day 30: Job change
htma chat
> I got a new job! I now work at AI Labs as Principal Engineer
> exit

# The system maintains both current and historical information
htma query "where do I work?"
# Shows: AI Labs (current)

htma query "where did I work before?"
# Shows: TechCorp (historical)
```

## Tips

1. **Regular Consolidation**: Run `htma consolidate` periodically to organize memories
2. **Export Regularly**: Use `htma export` to backup important memories
3. **Memory Status**: Check `htma status` to monitor memory growth
4. **Targeted Queries**: Use semantic/episodic filters to find specific types of information
5. **Demo First**: Try `python scripts/demo.py` to understand capabilities

## Architecture Notes

The CLI demonstrates HTMA's tri-memory architecture:

- **Working Memory**: In-context (not persisted, 8K token limit)
- **Semantic Memory**: Facts and entities (SQLite + embeddings)
- **Episodic Memory**: Experiences in hierarchy (L0=raw, L1=summaries, L2+=abstractions)

Memory flows:
1. User input → Working memory
2. Retrieve relevant context → Inject into working memory
3. Generate response
4. Store interaction → Curator evaluates → Create entities/facts/episodes
5. Link generator connects to existing memories
6. Consolidation creates abstractions (background/on-demand)

## Contributing

When adding new CLI commands:

1. Add command function with `@app.command()` decorator
2. Implement async version with `_command_name` prefix
3. Use Rich for output formatting
4. Add error handling
5. Document in this README

When adding demo scenarios:

1. Create scenario method in `HTMADemo` class
2. Follow pattern: intro panel → interactions → completion message
3. Add to scenario menu
4. Update README

## License

MIT License - See LICENSE file for details
