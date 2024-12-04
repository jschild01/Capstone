
import os
from py2neo import Graph


# Set up a connection to your Neo4j instance (use environment variables for security)
neo4j_url = os.getenv("NEO4J_URL", "neo4j+s://767b9d2a.databases.neo4j.io")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "JdfbXB1ZSMap1D1L91VmahNhwUAntGAk6PQ8iUXSVU0")
graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))


try:
    graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))
    result = graph.run("MATCH (n) RETURN n LIMIT 1").data()
    print("Connection successful, data:", result)
except Exception as e:
    print("Connection failed:", str(e))