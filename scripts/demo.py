#!/usr/bin/env python3
"""HTMA Guided Demo Script.

This script provides an interactive, guided demonstration of HTMA capabilities
through a series of scenarios that showcase different features.

Scenarios:
1. Introduction and basic conversation
2. Teaching facts and seeing retrieval
3. Time-based changes (bi-temporal reasoning)
4. Pattern emergence over multiple conversations
5. Consolidation and abstraction
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htma.agent.agent import HTMAAgent
from htma.consolidation.abstraction import AbstractionGenerator
from htma.consolidation.engine import ConsolidationEngine
from htma.consolidation.patterns import PatternDetector
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

console = Console()

# Configuration
DEFAULT_DATA_DIR = Path.home() / ".htma_demo"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "htma.db"
DEFAULT_CHROMA_PATH = DEFAULT_DATA_DIR / "chroma"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_REASONER_MODEL = "llama3:8b"
DEFAULT_CURATOR_MODEL = "mistral:7b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class HTMADemo:
    """HTMA guided demo orchestrator."""

    def __init__(self):
        self.agent: Optional[HTMAAgent] = None
        self.memory: Optional[MemoryInterface] = None
        self.consolidation: Optional[ConsolidationEngine] = None
        self.llm: Optional[OllamaClient] = None
        self.conv_id: Optional[str] = None

    async def initialize(self):
        """Initialize the HTMA system."""
        console.print("\n[bold cyan]Initializing HTMA Demo System...[/bold cyan]\n")

        # Ensure data directory exists
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize LLM client
        self.llm = OllamaClient(base_url=DEFAULT_OLLAMA_URL)

        # Check Ollama connection
        with console.status("[bold green]Checking Ollama connection..."):
            if not await self.llm.health_check():
                console.print(
                    f"[red]Error: Cannot connect to Ollama at {DEFAULT_OLLAMA_URL}[/red]\n"
                    "Please ensure Ollama is running and try again."
                )
                sys.exit(1)

        console.print("[green]✓ Connected to Ollama[/green]")

        # Initialize storage
        with console.status("[bold green]Initializing storage..."):
            sqlite = SQLiteStorage(db_path=str(DEFAULT_DB_PATH))
            await sqlite.initialize()

            chroma = ChromaStorage(
                persist_path=str(DEFAULT_CHROMA_PATH),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
            )
            await chroma.initialize()

        console.print("[green]✓ Storage initialized[/green]")

        # Initialize memory stores
        with console.status("[bold green]Setting up memory systems..."):
            working_config = WorkingMemoryConfig(
                max_tokens=8000,
                pressure_threshold=0.8,
                summarization_model=DEFAULT_CURATOR_MODEL,
            )
            working = WorkingMemory(config=working_config, llm=self.llm)
            semantic = SemanticMemory(sqlite=sqlite, chroma=chroma)
            episodic = EpisodicMemory(sqlite=sqlite, chroma=chroma)

            # Initialize extractors
            entity_extractor = EntityExtractor(llm=self.llm, model=DEFAULT_CURATOR_MODEL)
            fact_extractor = FactExtractor(llm=self.llm, model=DEFAULT_CURATOR_MODEL)
            link_generator = LinkGenerator(
                llm=self.llm, model=DEFAULT_CURATOR_MODEL, episodic=episodic
            )

            # Initialize curator WITH extractors injected
            curator = MemoryCurator(
                llm=self.llm,
                model=DEFAULT_CURATOR_MODEL,
                entity_extractor=entity_extractor,
                fact_extractor=fact_extractor,
                link_generator=link_generator,
            )

            # Initialize memory interface (4 parameters only)
            self.memory = MemoryInterface(
                working=working,
                semantic=semantic,
                episodic=episodic,
                curator=curator,
            )

            # Initialize agent
            agent_config = AgentConfig(
                reasoner_model=DEFAULT_REASONER_MODEL,
                auto_store_interactions=True,
            )
            self.agent = HTMAAgent(llm=self.llm, memory=self.memory, config=agent_config)

            # Initialize abstraction and pattern components
            abstraction_generator = AbstractionGenerator(
                llm=self.llm,
                model=DEFAULT_CURATOR_MODEL,
                episodic=episodic,
            )
            pattern_detector = PatternDetector(
                llm=self.llm,
                model=DEFAULT_CURATOR_MODEL,
            )

            # Initialize consolidation engine with all required components
            self.consolidation = ConsolidationEngine(
                curator=curator,
                semantic=semantic,
                episodic=episodic,
                abstraction_generator=abstraction_generator,
                pattern_detector=pattern_detector,
            )

        console.print("[green]✓ Memory systems ready[/green]")
        console.print("\n[bold green]System initialized successfully![/bold green]\n")

    async def send_message(self, message: str, show_retrieval: bool = True) -> str:
        """Send a message to the agent and display the response."""
        console.print(f"\n[bold blue]You:[/bold blue] {message}")

        with console.status("[bold green]Thinking..."):
            response = await self.agent.process_message(message, conversation_id=self.conv_id)

        console.print(f"\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(response.message))

        # Show memory context if requested
        if show_retrieval and response.retrieved_context:
            episodes_count = response.retrieved_context.get("episodes_count", 0)
            facts_count = response.retrieved_context.get("facts_count", 0)

            if episodes_count > 0 or facts_count > 0:
                memory_info = Text()
                memory_info.append("💭 ", style="dim cyan")

                parts = []
                if episodes_count > 0:
                    parts.append(f"{episodes_count} memory episode(s)")
                if facts_count > 0:
                    parts.append(f"{facts_count} fact(s)")

                memory_info.append(
                    f"Retrieved: {', '.join(parts)}",
                    style="dim italic cyan",
                )

                console.print(memory_info)

        return response.message

    async def wait_for_continue(self):
        """Wait for user to continue."""
        console.print()
        Prompt.ask("[dim]Press Enter to continue[/dim]", default="")

    async def scenario_1_introduction(self):
        """Scenario 1: Introduction and basic conversation."""
        console.print(Panel.fit(
            "[bold cyan]Scenario 1: Introduction & Basic Conversation[/bold cyan]\n\n"
            "This scenario demonstrates basic interaction with the HTMA system.\n"
            "The agent will remember information from this conversation.",
            border_style="cyan",
        ))

        await self.wait_for_continue()

        # Start conversation
        self.conv_id = self.agent.start_conversation()

        # Basic conversation
        await self.send_message("Hi! My name is Alex and I'm a software engineer.")
        await asyncio.sleep(0.5)

        await self.send_message("I work on machine learning projects, particularly in natural language processing.")
        await asyncio.sleep(0.5)

        await self.send_message("I'm interested in learning more about memory systems for AI agents.")

        console.print("\n[green]✓ Scenario 1 complete![/green]")
        console.print("[dim]The agent has stored information about you in memory.[/dim]")
        await self.wait_for_continue()

    async def scenario_2_fact_retrieval(self):
        """Scenario 2: Teaching facts and seeing retrieval."""
        console.print(Panel.fit(
            "[bold cyan]Scenario 2: Teaching Facts & Memory Retrieval[/bold cyan]\n\n"
            "This scenario shows how the agent remembers facts you've told it\n"
            "and retrieves them in future conversations.",
            border_style="cyan",
        ))

        await self.wait_for_continue()

        # Teach some facts
        await self.send_message("I have two cats named Whiskers and Luna.")
        await asyncio.sleep(0.5)

        await self.send_message("I prefer Python over JavaScript for backend development.")
        await asyncio.sleep(0.5)

        # Give time for storage
        await asyncio.sleep(2)

        # Now ask to recall
        console.print("\n[bold yellow]Testing memory retrieval...[/bold yellow]")
        await self.send_message("What are the names of my cats?")
        await asyncio.sleep(0.5)

        await self.send_message("What programming languages do I prefer?")

        console.print("\n[green]✓ Scenario 2 complete![/green]")
        console.print(
            "[dim]Notice how the agent retrieved facts from memory to answer your questions.[/dim]"
        )
        await self.wait_for_continue()

    async def scenario_3_temporal_reasoning(self):
        """Scenario 3: Time-based changes (bi-temporal reasoning)."""
        console.print(Panel.fit(
            "[bold cyan]Scenario 3: Temporal Reasoning[/bold cyan]\n\n"
            "This scenario demonstrates how HTMA handles facts that change over time.\n"
            "The system maintains both current and historical information.",
            border_style="cyan",
        ))

        await self.wait_for_continue()

        # Initial fact
        await self.send_message("I currently work at TechCorp as a Senior Engineer.")
        await asyncio.sleep(1)

        # Update the fact
        console.print("\n[bold yellow]Simulating a job change...[/bold yellow]")
        await self.send_message("I just got a new job! I now work at AI Labs as a Principal Engineer.")
        await asyncio.sleep(1)

        # Query current state
        await self.send_message("Where do I work now?")

        console.print("\n[green]✓ Scenario 3 complete![/green]")
        console.print(
            "[dim]The system maintains both your current and previous employment information.[/dim]"
        )
        await self.wait_for_continue()

    async def scenario_4_pattern_emergence(self):
        """Scenario 4: Pattern emergence over multiple conversations."""
        console.print(Panel.fit(
            "[bold cyan]Scenario 4: Pattern Detection[/bold cyan]\n\n"
            "This scenario shows how HTMA detects recurring patterns in your behavior.\n"
            "We'll mention similar activities multiple times.",
            border_style="cyan",
        ))

        await self.wait_for_continue()

        # Create a pattern through repetition
        pattern_messages = [
            "I went for a morning run today. I usually run around 7 AM.",
            "Had my morning run again today. I really enjoy starting the day with exercise.",
            "Completed my morning run routine. It's become a daily habit.",
        ]

        for i, msg in enumerate(pattern_messages, 1):
            console.print(f"\n[dim]Pattern observation {i}/3...[/dim]")
            await self.send_message(msg, show_retrieval=False)
            await asyncio.sleep(0.5)

        # Give time for processing
        await asyncio.sleep(1)

        # Ask about the pattern
        console.print("\n[bold yellow]Testing pattern recognition...[/bold yellow]")
        await self.send_message("What's my morning routine?")

        console.print("\n[green]✓ Scenario 4 complete![/green]")
        console.print(
            "[dim]The agent recognized the pattern of morning runs from repeated mentions.[/dim]"
        )
        await self.wait_for_continue()

    async def scenario_5_consolidation(self):
        """Scenario 5: Memory consolidation and abstraction."""
        console.print(Panel.fit(
            "[bold cyan]Scenario 5: Memory Consolidation[/bold cyan]\n\n"
            "This scenario demonstrates the consolidation process where\n"
            "HTMA creates higher-level summaries and abstractions from individual memories.",
            border_style="cyan",
        ))

        await self.wait_for_continue()

        # Check current memory state
        console.print("\n[bold yellow]Current memory state:[/bold yellow]")

        with console.status("[green]Querying memory..."):
            # Get episode counts
            level_0 = await self.memory.episodic.get_recent(level=0, limit=100)
            level_1 = await self.memory.episodic.get_recent(level=1, limit=100)

        stats_table = Table(show_header=True, header_style="bold cyan")
        stats_table.add_column("Level", style="cyan")
        stats_table.add_column("Count", style="green", justify="right")
        stats_table.add_column("Description", style="yellow")

        stats_table.add_row("Level 0", str(len(level_0)), "Raw interactions")
        stats_table.add_row("Level 1", str(len(level_1)), "Summaries")

        console.print(stats_table)
        console.print()

        # Run consolidation
        if len(level_0) >= 3:
            console.print("[bold yellow]Running consolidation cycle...[/bold yellow]\n")

            with console.status("[green]Consolidating memories..."):
                report = await self.consolidation.run_cycle()

            # Show consolidation results
            console.print("[bold green]Consolidation Report:[/bold green]\n")

            report_table = Table(show_header=True, header_style="bold cyan")
            report_table.add_column("Metric", style="cyan")
            report_table.add_column("Value", style="green", justify="right")

            report_table.add_row("Abstractions Created", str(report.abstractions_created))
            report_table.add_row("Patterns Detected", str(report.patterns_detected))
            report_table.add_row("Links Strengthened", str(report.links_strengthened))
            report_table.add_row("Links Pruned", str(report.links_pruned))
            report_table.add_row("Processing Time", f"{report.processing_time:.2f}s")

            console.print(report_table)
        else:
            console.print(
                "[yellow]Not enough episodes for consolidation.[/yellow]\n"
                "[dim]In a real scenario, consolidation would run after accumulating more memories.[/dim]"
            )

        console.print("\n[green]✓ Scenario 5 complete![/green]")
        console.print(
            "[dim]Consolidation creates summaries and detects patterns across memories.[/dim]"
        )
        await self.wait_for_continue()

    async def run_demo(self):
        """Run the full guided demo."""
        # Show welcome screen
        console.print(Panel.fit(
            "[bold cyan]Welcome to the HTMA Interactive Demo![/bold cyan]\n\n"
            "This guided demonstration will walk you through the key features of\n"
            "Hierarchical-Temporal Memory Architecture (HTMA) for LLM agents.\n\n"
            "You'll experience:\n"
            "  • Memory-augmented conversations\n"
            "  • Fact storage and retrieval\n"
            "  • Temporal reasoning\n"
            "  • Pattern detection\n"
            "  • Memory consolidation",
            border_style="cyan",
            padding=(1, 2),
        ))

        console.print()

        # Initialize system
        await self.initialize()

        # Show scenario menu
        while True:
            console.print("\n[bold]Available Scenarios:[/bold]\n")

            scenarios = [
                ("Introduction & Basic Conversation", self.scenario_1_introduction),
                ("Teaching Facts & Memory Retrieval", self.scenario_2_fact_retrieval),
                ("Temporal Reasoning", self.scenario_3_temporal_reasoning),
                ("Pattern Detection", self.scenario_4_pattern_emergence),
                ("Memory Consolidation", self.scenario_5_consolidation),
            ]

            for i, (name, _) in enumerate(scenarios, 1):
                console.print(f"  {i}. {name}")

            console.print(f"  {len(scenarios) + 1}. Run all scenarios")
            console.print(f"  {len(scenarios) + 2}. Free chat")
            console.print(f"  0. Exit demo")

            console.print()
            choice = IntPrompt.ask(
                "Select a scenario",
                choices=[str(i) for i in range(len(scenarios) + 3)],
            )

            if choice == 0:
                break
            elif choice == len(scenarios) + 1:
                # Run all scenarios
                for i, (name, scenario_func) in enumerate(scenarios, 1):
                    await scenario_func()
            elif choice == len(scenarios) + 2:
                # Free chat
                await self.free_chat()
            elif 1 <= choice <= len(scenarios):
                # Run selected scenario
                await scenarios[choice - 1][1]()

        # End conversation
        if self.conv_id:
            await self.agent.end_conversation(self.conv_id)

        console.print("\n[bold cyan]Thank you for trying HTMA![/bold cyan]")
        console.print(
            f"\n[dim]Demo data is stored in: {DEFAULT_DATA_DIR}[/dim]\n"
            "[dim]You can clear it by deleting this directory.[/dim]\n"
        )

    async def free_chat(self):
        """Free chat mode."""
        console.print(Panel.fit(
            "[bold cyan]Free Chat Mode[/bold cyan]\n\n"
            "Chat freely with the agent. Type 'back' to return to the menu.",
            border_style="cyan",
        ))

        if not self.conv_id:
            self.conv_id = self.agent.start_conversation()

        while True:
            console.print()
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            if user_input.lower().strip() in ["back", "exit", "menu"]:
                break

            if not user_input.strip():
                continue

            await self.send_message(user_input)


async def main():
    """Main entry point."""
    demo = HTMADemo()
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
