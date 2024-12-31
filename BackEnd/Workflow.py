from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import MessagesState

from langgraph.graph import END, StateGraph, MessagesState
from RagPipeline import (
    embed_and_store_nodes,
    generate_query_embedding,
    find_most_similar_node,
    query_graph_for_non_null_properties,
    get_or_create_embeddings,
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq model
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=GROQ_API_KEY,
)

# Define the Info schema and state for handling messages
class Info(BaseModel):
    relationship: str = Field(description="Name of the relationship.")


class State(MessagesState):
    query: str
    response: str
    relationship: str


# Decision-making process to identify the relationship
def decision(state: State):
    "Determine the user's intent and corresponding graph relationship."
    parser = JsonOutputParser(pydantic_object=Info)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant helping to analyze user queries related to airports and airlines. Based on the query, your task is to decide which relationship to use between nodes in the graph database. The relationships available are:

                    1. **OPERATES**: Represents the relationship between an airline and the routes it works in. This is only used to gather information about an airline.
                    2. **SERVES**: Represents the relationship between an airport and the airlines that serve it.
                    3. **CONNECTED_TO**: Represents the relationship between source and destination airports, showing which airports are connected to each other, or a city to another city, or a country to another country.
                    4. **LOCATED_IN**: Represents the relationship between an airport and the city it's located in, as well as the relationship between a city and the country it belongs to.

                Based on the user's query, return one of the following relationships: `OPERATES`, `SERVES`, `CONNECTED_TO`, `LOCATED_IN`.
                Answer only with one of the above relationships.
                
                ### Example Queries:
                1. **Query**: "Which routes does Airline X operate?"
                - **Answer**: Use the `OPERATES` relationship because the query is asking about the routes associated with an airline.

                2. **Query**: "Which airlines serve Goroka Airport?"
                - **Answer**: Use the `SERVES` relationship because the query is asking about airlines that serve a specific airport.

                3. **Query**: "What airports are connected to Heathrow?"
                - **Answer**: Use the `CONNECTED_TO` relationship because the query is asking about airports that are connected to Heathrow.

                4. **Query**: "Where is JFK Airport located?"
                - **Answer**: Use the `LOCATED_IN` relationship because the query is asking about the city or country where the airport is located.

                ### Your Task:
                Carefully analyze the user's query and return the most relevant relationship.
                
                {format_instructions}
                """,
            ),
            ("human", "{input}"),
        ]
    )

    info_chain = prompt | llm | parser

    try:
        # Process the query to determine the relationship
        json = info_chain.invoke(
            {
                "input": state["query"],
                "format_instructions": parser.get_format_instructions(),
            }
        )
        state["relationship"] = json["relationship"]
        return state
    except Exception as e:
        print(f"Error while determining relationship: {e}")
        return None


# Execute the pipeline to query the graph and format the response
def execute_pipeline(state: State):
    # Load or create embeddings (to avoid recalculating every time)
    embeddings = get_or_create_embeddings()
    embed_and_store_nodes(embeddings)  # Store embeddings into Neo4j

    # Process the query
    query = state["query"]
    query_embedding = generate_query_embedding(query)
    best_match = find_most_similar_node(query_embedding)

    # Fetch graph data based on the relationship
    result = query_graph_for_non_null_properties(best_match, state["relationship"])
    print("Result from graph query:", result)

    # Format the response using the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant responsible for rephrasing the information from the database to answer a user's query in a clear and conversational way.
                Given the information below, rephrase the response while ensuring all relevant details are retained.

                Information: {information}
                
                User Query: {input}
                
                Your task is to rephrase the information so that it is clear, concise, and answers the query. If there are multiple routes or pieces of data, make sure they are presented appropriately without omitting anything necessary.
                Remember that you are interacting directly with the user so answer without hesitation and do not say there is a rephrase ..
                Example:
                - For multiple routes: "This airline operates the following routes: NYC to LA, LA to SF, SF to DC."
                - If no relevant properties exist: "No relevant information was found."
                """
            ),
            ("human", ""),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    state["response"] = chain.invoke({"information": result, "input": query})
    return state

def get_graph():
    
    workflow = StateGraph(State)

    workflow.add_node("decision", decision)

    workflow.add_node("execute_pipeline", execute_pipeline)
    workflow.add_edge("decision", "execute_pipeline")
    workflow.add_edge("execute_pipeline", END)
   
    workflow.set_entry_point("decision")
    return workflow


# Main function
if __name__ == "__main__":
    # Create an initial state
    instance = State(
        query="Where is Mount Hagenn Kagamuga Airport?",
        relationship="",
        response=""
    )

    # Determine the relationship
    decision(instance)

    # Execute the pipeline
    execute_pipeline(instance)

    # Print the results
    print("Relationship:", instance["relationship"])
    print("Response:", instance["response"])
