import os
import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
Neo4jPass = os.getenv("neo4jpassword")
# Neo4j setup
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Neo4jPass"))

# Load CSV data using pandas
airlines_df = pd.read_csv('airlines.csv')
airports_df = pd.read_csv('airports.csv')
routes_df = pd.read_csv('routes.csv')

# Initialize SentenceTransformer model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =SentenceTransformer('all-MiniLM-L6-v2', device=device)

# JSON file for saving embeddings
EMBEDDINGS_FILE = "embeddings.json"

# Initialize GPT-2 model and tokenizer for human-readable response generation



# Function to save embeddings to a JSON file
def save_embeddings_to_json(data, filename):
    print(f"Saving embeddings to {filename}...")
    try:
        with open(filename, "w") as file:
            json.dump(data, file)
        print("Embeddings saved successfully.")
    except Exception as e:
        print(f"Error saving embeddings: {e}")


# Function to load embeddings from a JSON file
def load_embeddings_from_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


# Generate embeddings or load from JSON
def get_or_create_embeddings():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.getsize(EMBEDDINGS_FILE) > 0:
        print("Loading embeddings from JSON file...")
        return load_embeddings_from_json(EMBEDDINGS_FILE)
    else:
        print("Generating embeddings and saving to JSON file...")

        # Collect all texts to process in a batch
        airline_texts = airlines_df["Name"].dropna().tolist()
        airport_texts = airports_df["Name"].dropna().tolist()
        route_texts = [
            f"{row['SourceAirport']} to {row['DestAirport']}" 
            for _, row in routes_df.iterrows()
            if pd.notnull(row['SourceAirport']) and pd.notnull(row['DestAirport'])
        ]

        # Generate embeddings
        airline_embeddings = model.encode(airline_texts, batch_size=32, show_progress_bar=True).tolist()
        airport_embeddings = model.encode(airport_texts, batch_size=32, show_progress_bar=True).tolist()
        route_embeddings = model.encode(route_texts, batch_size=32, show_progress_bar=True).tolist()

        # Structure embeddings
        embeddings = {
            "airlines": dict(zip(airlines_df["AirlineID"], airline_embeddings)),
            "airports": dict(zip(airports_df["AirportID"], airport_embeddings)),
            "routes": dict(
                (f"{row['SourceAirportID']}_{row['DestAirportID']}", route_embeddings[idx])
                for idx, row in routes_df.iterrows()
                if pd.notnull(row['SourceAirportID']) and pd.notnull(row['DestAirportID'])
            ),
        }

        # Save to JSON
        save_embeddings_to_json(embeddings, EMBEDDINGS_FILE)
        return embeddings


# Embed and store nodes in Neo4j
# Embed and store nodes in Neo4j
# Embed and store nodes in Neo4j
def embed_and_store_nodes(embeddings):
    with driver.session() as session:
        # Store airline embeddings in batches
        batch_size = 100
        for i in range(0, len(embeddings["airlines"]), batch_size):
            batch = list(embeddings["airlines"].items())[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS data
                MATCH (a:Airline {AirlineID: data.AirlineID})
                WHERE a.embedding IS NOT NULL
                SET a.embedding = data.embedding
                """,
                batch=[{"AirlineID": airline_id, "embedding": embedding} for airline_id, embedding in batch],
            )

        # Store airport embeddings
        for i in range(0, len(embeddings["airports"]), batch_size):
            batch = list(embeddings["airports"].items())[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS data
                MATCH (ap:Airport {AirportID: data.AirportID})
                WHERE ap.embedding IS NOT NULL
                SET ap.embedding = data.embedding
                """,
                batch=[{"AirportID": airport_id, "embedding": embedding} for airport_id, embedding in batch],
            )

        # Store route embeddings
        for i in range(0, len(embeddings["routes"]), batch_size):
            batch = list(embeddings["routes"].items())[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS data
                MATCH (r:Route {SourceAirportID: data.SourceAirportID, DestAirportID: data.DestAirportID})
                WHERE r.embedding IS NOT NULL
                SET r.embedding = data.embedding
                """,
                batch=[
                    {"SourceAirportID": source_id, "DestAirportID": dest_id, "embedding": embedding}
                    for (source_id, dest_id), embedding in [
                        (route_id.split("_"), embedding) for route_id, embedding in batch
                    ]
                ],
            )




# Generate embedding for a query
def generate_query_embedding(query):
    return model.encode([query], show_progress_bar=False).tolist()[0]


# Generate human-readable response using GPT-2 based on the query and matched entity



# Find most similar node based on cosine similarity
def find_most_similar_node(query_embedding):
    with driver.session() as session:
        # Query for all nodes with embeddings
        result = session.run(
            """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN n.id AS node_id, n.Name AS node_name, n.embedding AS node_embedding
        """
        )

        highest_similarity = -1
        best_match = None

        # Reshape query_embedding to 2D
        query_embedding = np.array(query_embedding).reshape(1, -1)

        for record in result:
            node_embedding = record["node_embedding"]

            # Reshape node_embedding to 2D as well
            node_embedding = np.array(node_embedding).reshape(1, -1)

            similarity = cosine_similarity(query_embedding, node_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = record["node_name"]  # Store the name of the best matching node

        if best_match:
            return best_match
        else:
            return "No relevant match found."


def query_graph_for_non_null_properties(best_match, relationship):
    query = """
    MATCH (n {name: $best_match})-[r]->(related)
    WHERE type(r) = $relationship
    WITH related, keys(related) AS related_keys
    UNWIND related_keys AS key
    WITH related, key, coalesce(related[key], "N/A") AS value
    WHERE value IS NOT NULL AND value <> "N/A"
    RETURN key, value
    """
    
    with driver.session() as session:
        # Running the query and passing 'best_match' and 'relationship' as parameters
        result = session.run(query, best_match=best_match, relationship=relationship)
        
        # Convert result to a list of records
        records = list(result)
        print(f"Query result: {records}")
        
        # Dictionary to store key-value pairs, with each key having a list of values
        grouped_properties = {}
        
        for record in records:
            key = record["key"]
            value = record["value"]
            
            # Add the value to the list of values for the corresponding key
            if key not in grouped_properties:
                grouped_properties[key] = []
            grouped_properties[key].append(value)

        # Start the formatted result with the relationship name
        formatted_result = f"{relationship}: "
        
        # Iterate over the grouped properties and format them
        for key, values in grouped_properties.items():
            # Join the values with commas and wrap in square brackets
            formatted_result += f"[{key}: {', '.join(map(str, values))}] "
        
        print("-a-:", formatted_result)  # Debugging: Check the formatted result

    # Return the final formatted result
    return formatted_result



   






result = query_graph_for_non_null_properties("Airlines PNG", "OPERATES")
