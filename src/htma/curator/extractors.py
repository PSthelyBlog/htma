"""Entity and fact extraction from text using LLM.

This module provides extractors for identifying entities and relationships
in text, which are then used to populate semantic memory.
"""

import json
import logging
from pathlib import Path

from htma.core.exceptions import LLMResponseError
from htma.core.types import ExtractedEntity, ExtractedFact
from htma.llm.client import OllamaClient

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts" / "curator"


class EntityExtractor:
    """Extracts entities from text using LLM.

    Entity types supported:
    - person: Individual people
    - place: Locations, venues, addresses
    - organization: Companies, groups, institutions
    - concept: Abstract ideas, topics, subjects
    - object: Physical items, products, tools
    - event: Occurrences, meetings, activities
    - time: Temporal references (dates, times, periods)

    Attributes:
        llm: LLM client for extraction operations.
        model: Model name to use for extraction.
    """

    def __init__(self, llm: OllamaClient, model: str = "mistral:7b"):
        """Initialize entity extractor.

        Args:
            llm: LLM client for extraction operations.
            model: Model name to use (default: mistral:7b).
        """
        self.llm = llm
        self.model = model
        logger.info(f"Initialized EntityExtractor with model {model}")

    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from file.

        Args:
            template_name: Name of the template file.

        Returns:
            Template content as string.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        template_path = PROMPTS_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text()

    async def extract(self, text: str, context: str = "") -> list[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: The text to extract entities from.
            context: Additional context for extraction (e.g., conversation history).

        Returns:
            List of extracted entities with metadata.

        Raises:
            LLMResponseError: If LLM fails to return valid JSON or returns invalid data.

        Example:
            >>> extractor = EntityExtractor(llm)
            >>> entities = await extractor.extract("Alice works at Google in Mountain View")
            >>> for entity in entities:
            ...     print(f"{entity.name} ({entity.entity_type})")
            Alice (person)
            Google (organization)
            Mountain View (place)
        """
        logger.debug("Extracting entities from text")

        # Handle edge cases
        if not text or not text.strip():
            logger.warning("Empty text provided for entity extraction")
            return []

        # Load and format prompt
        try:
            prompt_template = self._load_prompt_template("entity_extraction.txt")
            prompt = prompt_template.format(context=context, text=text)
        except Exception as e:
            logger.error(f"Failed to load/format prompt template: {e}")
            raise LLMResponseError(f"Failed to load prompt template: {e}") from e

        # Get LLM extraction
        try:
            response = await self.llm.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=1000,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMResponseError(f"Failed to generate entity extraction: {e}") from e

        # Parse JSON response
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Find JSON array or object in response
            start_idx = max(response.find("["), response.find("{"))
            if response.find("[") != -1 and (
                response.find("{") == -1 or response.find("[") < response.find("{")
            ):
                # Array comes first
                start_idx = response.find("[")
                end_idx = response.rfind("]") + 1
            else:
                # Object comes first or no array
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Handle both array and object formats
            if isinstance(data, dict):
                # Check if it's wrapped in an "entities" key
                if "entities" in data:
                    entities_data = data["entities"]
                else:
                    # Single entity returned as object
                    entities_data = [data]
            elif isinstance(data, list):
                entities_data = data
            else:
                raise ValueError(f"Unexpected JSON format: {type(data)}")

            # Parse entities
            entities = []
            for entity_data in entities_data:
                try:
                    entity = ExtractedEntity(
                        name=entity_data["name"],
                        entity_type=entity_data["type"],
                        mentions=entity_data.get("mentions", []),
                        confidence=float(entity_data.get("confidence", 1.0)),
                        metadata=entity_data.get("metadata", {}),
                    )
                    entities.append(entity)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid entity data: {e}\n{entity_data}")
                    continue

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")
            raise LLMResponseError(
                f"LLM returned invalid JSON response: {e}\nResponse: {response[:200]}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error parsing entity extraction: {e}")
            raise LLMResponseError(f"Failed to parse entity extraction: {e}") from e


class FactExtractor:
    """Extracts facts/relationships from text using LLM.

    Common predicates:
    - is_a: Entity type/classification
    - has_property: Property or attribute
    - located_in: Location relationship
    - works_at: Employment relationship
    - owns: Ownership
    - prefers: Preference
    - said: Statement attribution
    - believes: Belief attribution
    - happened_at: Event timing

    Attributes:
        llm: LLM client for extraction operations.
        model: Model name to use for extraction.
    """

    def __init__(self, llm: OllamaClient, model: str = "mistral:7b"):
        """Initialize fact extractor.

        Args:
            llm: LLM client for extraction operations.
            model: Model name to use (default: mistral:7b).
        """
        self.llm = llm
        self.model = model
        logger.info(f"Initialized FactExtractor with model {model}")

    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from file.

        Args:
            template_name: Name of the template file.

        Returns:
            Template content as string.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        template_path = PROMPTS_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text()

    async def extract(
        self, text: str, entities: list[ExtractedEntity], context: str = ""
    ) -> list[ExtractedFact]:
        """Extract facts/relationships from text given entities.

        Args:
            text: The text to extract facts from.
            entities: Entities to use for fact extraction.
            context: Additional context for extraction.

        Returns:
            List of extracted facts with metadata.

        Raises:
            LLMResponseError: If LLM fails to return valid JSON or returns invalid data.

        Example:
            >>> extractor = FactExtractor(llm)
            >>> entities = [
            ...     ExtractedEntity(name="Alice", entity_type="person"),
            ...     ExtractedEntity(name="Google", entity_type="organization")
            ... ]
            >>> facts = await extractor.extract("Alice works at Google", entities)
            >>> for fact in facts:
            ...     print(f"{fact.subject} {fact.predicate} {fact.object_entity or fact.object_value}")
            Alice works_at Google
        """
        logger.debug("Extracting facts from text")

        # Handle edge cases
        if not text or not text.strip():
            logger.warning("Empty text provided for fact extraction")
            return []

        if not entities:
            logger.debug("No entities provided for fact extraction")
            return []

        # Format entities for prompt
        entity_list = "\n".join(
            [f"- {e.name} ({e.entity_type})" for e in entities]
        )

        # Load and format prompt
        try:
            prompt_template = self._load_prompt_template("fact_extraction.txt")
            prompt = prompt_template.format(
                context=context, text=text, entities=entity_list
            )
        except Exception as e:
            logger.error(f"Failed to load/format prompt template: {e}")
            raise LLMResponseError(f"Failed to load prompt template: {e}") from e

        # Get LLM extraction
        try:
            response = await self.llm.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=1500,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMResponseError(f"Failed to generate fact extraction: {e}") from e

        # Parse JSON response
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Find JSON array or object in response
            start_idx = max(response.find("["), response.find("{"))
            if response.find("[") != -1 and (
                response.find("{") == -1 or response.find("[") < response.find("{")
            ):
                # Array comes first
                start_idx = response.find("[")
                end_idx = response.rfind("]") + 1
            else:
                # Object comes first or no array
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Handle both array and object formats
            if isinstance(data, dict):
                # Check if it's wrapped in a "facts" key
                if "facts" in data:
                    facts_data = data["facts"]
                else:
                    # Single fact returned as object
                    facts_data = [data]
            elif isinstance(data, list):
                facts_data = data
            else:
                raise ValueError(f"Unexpected JSON format: {type(data)}")

            # Parse facts
            facts = []
            for fact_data in facts_data:
                try:
                    fact = ExtractedFact(
                        subject=fact_data["subject"],
                        predicate=fact_data["predicate"],
                        object_entity=fact_data.get("object_entity"),
                        object_value=fact_data.get("object_value"),
                        temporal_marker=fact_data.get("temporal_marker"),
                        confidence=float(fact_data.get("confidence", 1.0)),
                        source_text=fact_data.get("source_text", ""),
                        metadata=fact_data.get("metadata", {}),
                    )
                    facts.append(fact)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid fact data: {e}\n{fact_data}")
                    continue

            logger.info(f"Extracted {len(facts)} facts from text")
            return facts

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")
            raise LLMResponseError(
                f"LLM returned invalid JSON response: {e}\nResponse: {response[:200]}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error parsing fact extraction: {e}")
            raise LLMResponseError(f"Failed to parse fact extraction: {e}") from e
