import redis
from neo4j import GraphDatabase
import chromadb
import json
import logging
import yaml
from datetime import datetime

class MemoryManager:
    def __init__(self):
        with open('aios/config/config.yaml', 'r') as f:
            self.cfg = yaml.safe_load(f)

        # 1. Short Term (Redis)
        self.redis = redis.Redis(
            host=self.cfg['memory']['redis_host'], 
            port=self.cfg['memory']['redis_port'], 
            db=0
        )

        # 2. Long Term Graph (Neo4j)
        self.neo4j = GraphDatabase.driver(
            self.cfg['memory']['neo4j_uri'], 
            auth=(self.cfg['memory']['neo4j_user'], self.cfg['memory']['neo4j_pass'])
        )

        # 3. Semantic Vector (Chroma)
        self.chroma = chromadb.PersistentClient(path="./root/db/chroma")
        self.vector_col = self.chroma.get_or_create_collection("os1_knowledge")

    def add_short_term(self, key, value):
        self.redis.setex(key, 3600, json.dumps(value)) # Expire in 1 hour

    def add_episodic_memory(self, user_input, agent_response, emotion):
        """Stores interaction in the Knowledge Graph"""
        query = """
        MERGE (u:User {name: 'Primary'})
        CREATE (i:Interaction {
            input: $inp, 
            response: $resp, 
            emotion: $emo, 
            timestamp: datetime()
        })
        MERGE (u)-[:EXPERIENCED]->(i)
        """
        with self.neo4j.session() as session:
            session.run(query, inp=user_input, resp=agent_response, emo=emotion)

    def retrieve_context(self, text_query):
        """Vector search for relevant past facts"""
        results = self.vector_col.query(query_texts=[text_query], n_results=3)
        if results['documents']:
            return " ".join(results['documents'][0])
        return ""