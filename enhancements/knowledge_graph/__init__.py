"""Compatibility imports for the knowledge graph module."""

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.entity_extractor import EntityExtractor
from src.jsjb.knowledge.query import GraphQueryEngine

__all__ = ["KnowledgeGraphManager", "EntityExtractor", "GraphQueryEngine"]
