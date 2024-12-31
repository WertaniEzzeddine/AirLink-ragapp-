import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.graph import MessagesState
from Workflow import get_graph  # Import your workflow graph logic

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class UserQuery(BaseModel):
    query: str

# TravelState Class
class State(MessagesState):
    query: str
    response: str
    relationship: str




# Process Query Function
def process_query(user_query: str):
    try:
        state = State(query=user_query, relationship="", response="")
        logging.debug(f"State before workflow: {state}")

        workflow = get_graph()
        app = workflow.compile()
        result = app.invoke(state)
        return result
    except Exception as e:
        logging.error(f"Error in process_query: {str(e)}")
        raise e


# FastAPI Endpoint
@app.post("/chatbot/")
async def chatbot(user_query: UserQuery):
    try:
        result = process_query(user_query.query)
        return {"response": result}
    except Exception as e:
        logging.error(f"Error processing the query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

