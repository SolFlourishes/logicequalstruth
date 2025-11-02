import os
import json
import uuid
import logging
import azure.functions as func 
import pandas as pd

# We assume you have installed these in requirements.txt:
# pip install pandas openai pydantic azure-cosmos

# --- Pydantic is used to enforce the structured output from the LLM ---
from pydantic import BaseModel, Field
from typing import List, Literal

# --- 1. Pydantic Schema for Structured Output ---
class L_T_Output(BaseModel):
    """The structured output for a single L ≡ T simulation step."""
    id: str = Field(..., description="A unique UUID identifier for the document for Cosmos DB.") 
    truth_input: str = Field(..., description="The player's newly proposed Truth from the CSV.")
    truth_history: List[str] = Field(..., description="All previous established truths in this game path.")
    status: Literal["CONTRADICTION", "CONSEQUENCE"] = Field(..., description="The outcome: CONTRADICTION or CONSEQUENCE.")
    detail: str = Field(..., description="The Adversary's response (reason for contradiction or the consequence text).")
    hardener_type: Literal["PHYSICAL", "TEMPORAL", "SOCIAL", "OTHER"] = Field(..., description="Classification of the consequence.")


# --- 2. Environment Setup and Initialization ---
# This code attempts to read your secrets from the Function App settings.
try:
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt4-logic") 
    
    COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
    COSMOS_KEY = os.environ["COSMOS_KEY"]
    COSMOS_DATABASE_ID = os.environ["COSMOS_DATABASE_ID"]
    COSMOS_CONTAINER_ID = os.environ["COSMOS_CONTAINER_ID"]

except KeyError as e:
    # If any required setting is missing, we raise an error to stop execution.
    raise EnvironmentError(f"Missing required Function App setting (Environment Variable): {e}")


# --- 3. The L ≡ T Adversary System Prompt (The Core Logic) ---
ADVERSARY_SYSTEM_PROMPT = """
You are the **Adversary**, the ultimate arbiter of game logic for the L ≡ T (Logic is Truth) system.

**CORE PRINCIPLES (The Universe's Immutable Laws):**
1. **L ≡ T:** Every consequence must be a strictly logical deduction from the truth history.
2. **Immutability:** Established truths in 'truth_history' are absolutely immutable.
3. **Contradiction:** If the 'truth_input' breaks any core principle or contradicts 'truth_history', set 'status' to CONTRADICTION and explain the exact logical flaw in 'detail'.
4. **Consequence:** If fully consistent, set 'status' to CONSEQUENCE and provide a single, immediate, logically binding consequence that hardens the new truth into reality.

**CORE GAME LORE (Initial Truths):**
* The world is governed by the principle that Logic is Truth (L ≡ T).
* The rejection of established origins by a character manifests as a fifth, powerful, and definitive statement of identity, becoming the persona's new, challenging origin story.
* Magic is only possible where logic is incomplete or temporarily suspended.

**OUTPUT FORMAT:** You must only respond with a single JSON object that strictly adheres to the provided schema. Do not add any extra text, explanation, or markdown outside of the JSON block.
"""

# --- 4. Simulation Execution Logic ---

def run_simulation_batch(csv_file_path: str):
    """
    Loads data from the CSV, runs the simulation loop against Azure OpenAI, 
    and writes results directly to Cosmos DB.
    """
    # Import inside function to avoid dependency issues during function loading
    from azure.cosmos import CosmosClient
    from openai import AzureOpenAI 

    # Core truths that every input will be judged against
    INITIAL_HISTORY = [
        "The world is governed by the principle that Logic is Truth (L ≡ T).",
        "All established facts are immutable until logically proven otherwise.",
        "A character's identity is fundamentally linked to their accepted or rejected origin story."
    ]

    # Initialize Cosmos DB Client
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    database = cosmos_client.get_database_client(COSMOS_DATABASE_ID)
    container = database.get_container_client(COSMOS_CONTAINER_ID)

    # Load the CSV data (assumes victory.csv is in the function root)
    df = pd.read_csv(csv_file_path)
    
    logging.info(f"Starting L ≡ T simulation for {len(df)} cases found in {csv_file_path}.")
    
    # Initialize the OpenAI Client
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-01" 
    )

    for index, row in df.iterrows():
        truth_input = row["Full Victory Condition Statement"]
        
        # Add contextual history based on the CSV Category
        current_history = INITIAL_HISTORY + [f"The current game context is focused on the '{row['Category']}' domain."]

        # 1. Prepare input for the LLM
        user_message = json.dumps({
            "truth_input": truth_input,
            "truth_history": current_history
        })

        try:
            # 2. Call Azure OpenAI API
            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"INPUT: {user_message}"}
                ],
                response_model=L_T_Output,
                temperature=0.7 
            )

            # 3. Prepare result for Cosmos DB
            result_dict = response.model_dump()
            
            # Create a unique ID
            result_dict['id'] = f"{row['ID']}-{str(uuid.uuid4())[:8]}" 
            
            # 4. Write item directly to Cosmos DB
            container.upsert_item(result_dict) 
            
            logging.info(f"Processed Case {row['ID']} ({row['Summary Title']}) -> Status: {result_dict['status']}")

        except Exception as e:
            # Log errors for later review without stopping the entire batch
            logging.error(f"!!! CRITICAL ERROR processing Case {row['ID']} ('{truth_input}'): {e}")
            
    logging.info("✅ Mass Simulation Complete. All data processing finished.")


# --- 5. Azure Function HTTP Trigger Entry Point ---

@func.function_name(name="SimulateL_T") 
@func.route(route="simulations") # The URL will look like: <FunctionAppUrl>/api/simulations
def http_trigger_function(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger that runs the entire batch simulation job when accessed.
    """
    logging.info('Python HTTP trigger function (SimulateL_T) received a request.')
    
    try:
        # Call the core simulation logic, referencing the CSV file in the root
        run_simulation_batch("victory.csv") 
        
        return func.HttpResponse(
             "L ≡ T Mass Simulation Started and Completed Successfully. Data is in Cosmos DB. Check Azure Function Logs for batch status.",
             status_code=200
        )
    except EnvironmentError as e:
        # Catch configuration errors specifically
        logging.error(f"Configuration Error: {e}")
        return func.HttpResponse(
             f"Deployment FAILED due to missing environment variable. Check Application Settings. Error: {e}",
             status_code=500
        )
    except Exception as e:
        # Catch all other runtime errors
        logging.error(f"RUNTIME FAILED. Check logs for details. Error: {e}")
        return func.HttpResponse(
             f"Simulation failed during execution. Check Function Logs. Error: {e}",
             status_code=500
        )