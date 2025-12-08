"""HTMA CLI - Interactive demo and command-line interface.

This module provides a command-line interface for interacting with the HTMA
memory system. It includes commands for chat, memory queries, consolidation,
and system management.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from htma.agent.agent import HTMAAgent
from htma.consolidation.engine import ConsolidationEngine
from htma.core.types import AgentConfig
from htma.curator.curator import MemoryCurator
from htma.curator.extractors import EntityExtractor, FactExtractor
from htma.curator.linkers import LinkGenerator
from htma.llm.client import OllamaClient
from htma.memory.episodic import EpisodicMemory
from htma.memory.interface import MemoryInterface
from htma.memory.semantic import SemanticMemory
from htma.memory.working import WorkingMemory, WorkingMemoryConfig
from htma.storage.chroma import ChromaStorage
from htma.storage.sqlite import SQLiteStorage

# Initialize Typer app and Rich console
app = typer.Typer(
    name="htma",
    help="HTMA - Hierarchical-Temporal Memory Architecture for LLM agents",
    add_completion=False,
)
console = Console()


# Global configuration
DEFAULT_DATA_DIR = Path.home() / ".htma"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "htma.db"
DEFAULT_CHROMA_PATH = DEFAULT_DATA_DIR / "chroma"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_REASONER_MODEL = "llama3:8b"
DEFAULT_CURATOR_MODEL = "mistral:7b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def get_data_dir() -> Path:
    """Get the data directory, creating it if it doesn't exist."""
    data_dir = DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


async def initialize_system(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    reasoner_model: str = DEFAULT_REASONER_MODEL,
    curator_model: str = DEFAULT_CURATOR_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[HTMAAgent, MemoryInterface, ConsolidationEngine, OllamaClient]:
    """Initialize the HTMA system components.

    Returns:
        Tuple of (agent, memory_interface, consolidation_engine, llm_client)
    """
    # Ensure data directory exists
    get_data_dir()

    # Initialize LLM client
    llm = OllamaClient(base_url=ollama_url)

    # Check Ollama connection
    if not await llm.health_check():
        console.print(
            "[red]Error: Cannot connect to Ollama.[/red]\n"
            f"Please ensure Ollama is running at {ollama_url}"
        )
        raise typer.Exit(1)

    # Initialize storage
    sqlite = SQLiteStorage(db_path=str(DEFAULT_DB_PATH))
    await sqlite.initialize()

    chroma = ChromaStorage(
        persist_path=str(DEFAULT_CHROMA_PATH),
        embedding_model=embedding_model,
    )
    await chroma.initialize()

    # Initialize memory stores
    working_config = WorkingMemoryConfig(
        max_tokens=8000,
        pressure_threshold=0.8,
        summarization_model=curator_model,
    )
    working = WorkingMemory(config=working_config, llm=llm)
    semantic = SemanticMemory(sqlite=sqlite, chroma=chroma)
    episodic = EpisodicMemory(sqlite=sqlite, chroma=chroma)

    # Initialize curator
    curator = MemoryCurator(llm=llm, model=curator_model)
    entity_extractor = EntityExtractor(llm=llm, model=curator_model)
    fact_extractor = FactExtractor(llm=llm, model=curator_model)
    link_generator = LinkGenerator(llm=llm, model=curator_model, episodic=episodic)

    # Initialize memory interface
    memory_interface = MemoryInterface(
        working=working,
        semantic=semantic,
        episodic=episodic,
        curator=curator,
        entity_extractor=entity_extractor,
        fact_extractor=fact_extractor,
        link_generator=link_generator,
    )

    # Initialize agent
    agent_config = AgentConfig(
        reasoner_model=reasoner_model,
        auto_store_interactions=True,
    )
    agent = HTMAAgent(llm=llm, memory=memory_interface, config=agent_config)

    # Initialize consolidation engine
    consolidation_engine = ConsolidationEngine(
        curator=curator,
        semantic=semantic,
        episodic=episodic,
    )

    return agent, memory_interface, consolidation_engine, llm


@app.command()
def chat(
    ollama_url: str = typer.Option(
        DEFAULT_OLLAMA_URL, "--ollama-url", help="Ollama API URL"
    ),
    reasoner_model: str = typer.Option(
        DEFAULT_REASONER_MODEL, "--reasoner", help="Model for reasoning (LLM₁)"
    ),
    curator_model: str = typer.Option(
        DEFAULT_CURATOR_MODEL, "--curator", help="Model for memory curation (LLM₂)"
    ),
):
    """Start an interactive chat session with memory augmentation."""
    asyncio.run(_chat(ollama_url, reasoner_model, curator_model))


async def _chat(ollama_url: str, reasoner_model: str, curator_model: str):
    """Async implementation of chat command."""
    console.print(Panel.fit(
        "[bold cyan]HTMA Interactive Chat[/bold cyan]\n"
        "Chat with an AI assistant enhanced by hierarchical-temporal memory.\n"
        "Type 'exit', 'quit', or 'bye' to end the conversation.",
        border_style="cyan",
    ))

    # Initialize system with progress indicator
    with console.status("[bold green]Initializing HTMA system..."):
        try:
            agent, memory, consolidation, llm = await initialize_system(
                ollama_url, reasoner_model, curator_model
            )
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            raise typer.Exit(1)

    console.print("[green]System initialized![/green]\n")

    # Start conversation
    conv_id = agent.start_conversation()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            # Check for exit commands
            if user_input.lower().strip() in ["exit", "quit", "bye", "q"]:
                console.print("\n[cyan]Goodbye! Your memories have been saved.[/cyan]")
                await agent.end_conversation(conv_id)
                break

            if not user_input.strip():
                continue

            # Process message with spinner
            with console.status("[bold green]Thinking..."):
                response = await agent.process_message(
                    user_input, conversation_id=conv_id
                )

            # Display assistant response
            console.print(f"\n[bold green]Assistant[/bold green]")
            console.print(Markdown(response.message))

            # Show memory context if available
            if response.retrieved_context:
                episodes_count = response.retrieved_context.get("episodes_count", 0)
                facts_count = response.retrieved_context.get("facts_count", 0)

                if episodes_count > 0 or facts_count > 0:
                    memory_info = Text()
                    memory_info.append("💭 ", style="dim")

                    if episodes_count > 0:
                        memory_info.append(
                            f"Retrieved {episodes_count} memory episode(s)",
                            style="dim italic",
                        )

                    if facts_count > 0:
                        if episodes_count > 0:
                            memory_info.append(", ", style="dim")
                        memory_info.append(
                            f"{facts_count} fact(s)",
                            style="dim italic",
                        )

                    console.print(memory_info)

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Use 'exit' to quit properly.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text to search memory"),
    semantic: bool = typer.Option(True, "--semantic/--no-semantic", help="Include semantic memory"),
    episodic: bool = typer.Option(True, "--episodic/--no-episodic", help="Include episodic memory"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results to return"),
):
    """Query memory directly without starting a conversation."""
    asyncio.run(_query(text, semantic, episodic, limit))


async def _query(text: str, include_semantic: bool, include_episodic: bool, limit: int):
    """Async implementation of query command."""
    console.print(f"\n[bold]Querying memory:[/bold] {text}\n")

    # Initialize system
    with console.status("[bold green]Initializing HTMA system..."):
        try:
            agent, memory, _, _ = await initialize_system()
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            raise typer.Exit(1)

    # Execute query
    with console.status("[bold green]Searching memory..."):
        result = await memory.query(
            query=text,
            include_semantic=include_semantic,
            include_episodic=include_episodic,
            limit=limit,
        )

    # Display results
    if not result.facts and not result.episodes:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Show facts
    if result.facts:
        console.print(f"\n[bold cyan]Semantic Facts ({len(result.facts)}):[/bold cyan]")

        facts_table = Table(show_header=True, header_style="bold cyan")
        facts_table.add_column("Subject", style="green")
        facts_table.add_column("Predicate", style="yellow")
        facts_table.add_column("Object", style="blue")
        facts_table.add_column("Confidence", style="magenta")

        for fact in result.facts[:limit]:
            facts_table.add_row(
                fact.subject_id[:20] + "..." if len(fact.subject_id) > 20 else fact.subject_id,
                fact.predicate,
                str(fact.object_value or fact.object_id)[:40] + "..." if (fact.object_value or fact.object_id) and len(str(fact.object_value or fact.object_id)) > 40 else str(fact.object_value or fact.object_id or ""),
                f"{fact.confidence:.2f}",
            )

        console.print(facts_table)

    # Show episodes
    if result.episodes:
        console.print(f"\n[bold cyan]Episodic Memories ({len(result.episodes)}):[/bold cyan]\n")

        for i, episode in enumerate(result.episodes[:limit], 1):
            relevance = result.relevance_scores.get(episode.id, 0.0)

            panel_content = []
            panel_content.append(f"[bold]Content:[/bold] {episode.content[:200]}")
            if len(episode.content) > 200:
                panel_content.append("...")

            if episode.summary:
                panel_content.append(f"\n[dim]Summary:[/dim] {episode.summary[:150]}")

            if episode.keywords:
                keywords_str = ", ".join(episode.keywords[:5])
                panel_content.append(f"\n[dim]Keywords:[/dim] {keywords_str}")

            panel_content.append(
                f"\n[dim]Level: {episode.level} | "
                f"Salience: {episode.salience:.2f} | "
                f"Relevance: {relevance:.2f}[/dim]"
            )

            console.print(Panel(
                "\n".join(panel_content),
                title=f"Episode {i}",
                border_style="cyan",
            ))


@app.command()
def consolidate(
    force: bool = typer.Option(False, "--force", "-f", help="Force consolidation even if not needed"),
):
    """Trigger memory consolidation cycle."""
    asyncio.run(_consolidate(force))


async def _consolidate(force: bool):
    """Async implementation of consolidate command."""
    console.print("\n[bold]Memory Consolidation[/bold]\n")

    # Initialize system
    with console.status("[bold green]Initializing HTMA system..."):
        try:
            _, _, consolidation, _ = await initialize_system()
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            raise typer.Exit(1)

    # Check if consolidation is needed
    if not force:
        should_run = await consolidation.should_run()
        if not should_run:
            console.print(
                "[yellow]Consolidation not needed yet.[/yellow]\n"
                "Use --force to run anyway."
            )
            return

    # Run consolidation with progress
    console.print("[green]Running consolidation cycle...[/green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Consolidating memories...", total=None)

        report = await consolidation.run_cycle()

        progress.update(task, completed=True)

    # Display report
    console.print("\n[bold green]Consolidation Complete![/bold green]\n")

    report_table = Table(show_header=True, header_style="bold cyan")
    report_table.add_column("Metric", style="cyan")
    report_table.add_column("Value", style="green", justify="right")

    report_table.add_row("Abstractions Created", str(report.abstractions_created))
    report_table.add_row("Patterns Detected", str(report.patterns_detected))
    report_table.add_row("Patterns Strengthened", str(report.patterns_strengthened))
    report_table.add_row("Links Strengthened", str(report.links_strengthened))
    report_table.add_row("Links Pruned", str(report.links_pruned))
    report_table.add_row("Episodes Pruned", str(report.episodes_pruned))
    report_table.add_row("Processing Time", f"{report.processing_time:.2f}s")

    console.print(report_table)


@app.command()
def status():
    """Show memory system statistics."""
    asyncio.run(_status())


async def _status():
    """Async implementation of status command."""
    console.print("\n[bold]HTMA Memory System Status[/bold]\n")

    # Initialize system
    with console.status("[bold green]Initializing HTMA system..."):
        try:
            agent, memory, consolidation, llm = await initialize_system()
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            raise typer.Exit(1)

    # Gather statistics
    with console.status("[bold green]Gathering statistics..."):
        # Get episode counts by level
        level_0_episodes = await memory.episodic.get_recent(level=0, limit=1000)
        level_1_episodes = await memory.episodic.get_recent(level=1, limit=1000)
        level_2_episodes = await memory.episodic.get_recent(level=2, limit=1000)

        # Get total episodes
        all_episodes = level_0_episodes + level_1_episodes + level_2_episodes

        # Working memory stats
        working_tokens = memory.working.current_tokens
        working_utilization = memory.working.utilization

        # Search for all entities (rough count)
        all_entities = await memory.semantic.sqlite.fetch_all("SELECT COUNT(*) as count FROM entities")
        entity_count = all_entities[0]["count"] if all_entities else 0

        # Count facts
        all_facts = await memory.semantic.sqlite.fetch_all("SELECT COUNT(*) as count FROM facts")
        fact_count = all_facts[0]["count"] if all_facts else 0

    # Display statistics
    # System info
    system_table = Table(title="System Information", show_header=False)
    system_table.add_column("Property", style="cyan")
    system_table.add_column("Value", style="green")

    system_table.add_row("Data Directory", str(DEFAULT_DATA_DIR))
    system_table.add_row("Database", str(DEFAULT_DB_PATH))
    system_table.add_row("Ollama URL", DEFAULT_OLLAMA_URL)

    console.print(system_table)
    console.print()

    # Memory statistics
    memory_table = Table(title="Memory Statistics", show_header=True, header_style="bold cyan")
    memory_table.add_column("Component", style="cyan")
    memory_table.add_column("Count", style="green", justify="right")
    memory_table.add_column("Details", style="yellow")

    memory_table.add_row(
        "Working Memory",
        f"{working_tokens} tokens",
        f"{working_utilization*100:.1f}% utilized",
    )
    memory_table.add_row("Entities", str(entity_count), "Semantic memory")
    memory_table.add_row("Facts", str(fact_count), "Semantic memory")
    memory_table.add_row("Episodes (L0)", str(len(level_0_episodes)), "Raw interactions")
    memory_table.add_row("Episodes (L1)", str(len(level_1_episodes)), "Summaries")
    memory_table.add_row("Episodes (L2+)", str(len(level_2_episodes)), "Abstractions")
    memory_table.add_row("[bold]Total Episodes[/bold]", f"[bold]{len(all_episodes)}[/bold]", "All levels")

    console.print(memory_table)
    console.print()

    # Consolidation info
    last_cycle = consolidation.last_cycle
    if last_cycle:
        console.print(f"[dim]Last consolidation: {last_cycle.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    else:
        console.print("[dim]No consolidation cycles run yet[/dim]")


@app.command()
def export(
    output: Path = typer.Option(
        None, "--output", "-o", help="Output file path (default: htma_export_TIMESTAMP.json)"
    ),
):
    """Export memory state to JSON file."""
    asyncio.run(_export(output))


async def _export(output: Optional[Path]):
    """Async implementation of export command."""
    # Generate default filename if not provided
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(f"htma_export_{timestamp}.json")

    console.print(f"\n[bold]Exporting memory to:[/bold] {output}\n")

    # Initialize system
    with console.status("[bold green]Initializing HTMA system..."):
        try:
            _, memory, _, _ = await initialize_system()
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            raise typer.Exit(1)

    # Export data
    with console.status("[bold green]Exporting memory..."):
        # Get all entities
        entities_data = await memory.semantic.sqlite.fetch_all("SELECT * FROM entities")

        # Get all facts
        facts_data = await memory.semantic.sqlite.fetch_all("SELECT * FROM facts")

        # Get all episodes
        episodes_data = await memory.episodic.sqlite.fetch_all("SELECT * FROM episodes")

        # Get all links
        links_data = await memory.episodic.sqlite.fetch_all("SELECT * FROM episode_links")

        # Build export structure
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "entities": entities_data,
            "facts": facts_data,
            "episodes": episodes_data,
            "links": links_data,
            "statistics": {
                "total_entities": len(entities_data),
                "total_facts": len(facts_data),
                "total_episodes": len(episodes_data),
                "total_links": len(links_data),
            },
        }

    # Write to file
    try:
        with open(output, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        console.print(f"[green]Export successful![/green]")
        console.print(f"\nExported:")
        console.print(f"  - {len(entities_data)} entities")
        console.print(f"  - {len(facts_data)} facts")
        console.print(f"  - {len(episodes_data)} episodes")
        console.print(f"  - {len(links_data)} links")
        console.print(f"\nFile: {output.absolute()}")
    except Exception as e:
        console.print(f"[red]Error writing export file: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Clear all memory (DANGEROUS!)."""
    asyncio.run(_reset(force))


async def _reset(force: bool):
    """Async implementation of reset command."""
    console.print("\n[bold red]WARNING: This will delete ALL memory![/bold red]\n")

    if not force:
        confirmed = Confirm.ask(
            "Are you sure you want to delete all entities, facts, and episodes?",
            default=False,
        )

        if not confirmed:
            console.print("[yellow]Reset cancelled.[/yellow]")
            return

    # Initialize system
    with console.status("[bold green]Initializing HTMA system..."):
        try:
            _, memory, _, _ = await initialize_system()
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            raise typer.Exit(1)

    # Delete all data
    with console.status("[bold red]Deleting all memory..."):
        # Clear SQLite tables
        await memory.semantic.sqlite.execute("DELETE FROM facts")
        await memory.semantic.sqlite.execute("DELETE FROM entities")
        await memory.episodic.sqlite.execute("DELETE FROM episode_links")
        await memory.episodic.sqlite.execute("DELETE FROM episodes")
        await memory.episodic.sqlite.execute("DELETE FROM retrieval_indices")

        # Clear ChromaDB collections
        try:
            await memory.episodic.chroma.episodes_collection.delete()
        except:
            pass

        try:
            await memory.semantic.chroma.entities_collection.delete()
        except:
            pass

        # Reinitialize collections
        await memory.episodic.chroma.initialize()
        await memory.semantic.chroma.initialize()

    console.print("[green]All memory has been cleared.[/green]")


@app.callback()
def callback():
    """
    HTMA - Hierarchical-Temporal Memory Architecture

    A memory system for LLM-based personal agents with:
    - Working memory with pressure management
    - Semantic memory (temporal knowledge graph)
    - Episodic memory (hierarchical experiences)
    - Autonomous memory curation and consolidation
    """
    pass


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
