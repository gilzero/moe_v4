# moe_v4.py
"""
This script implements a Multi-Agent Orchestration (MOE) system using various Large Language Models (LLMs)
to analyze a given query. It follows a structured workflow, gathering insights from different LLMs with
specific styles, analyzing consensus, generating visual representations, performing detailed analysis,
suggesting related questions, and conducting meta-analysis. The script is designed to be highly
configurable, robust, and easy to maintain, with asynchronous operations for improved performance.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Callable, List, Tuple, Any
import logging
from rich.console import Console
from rich.markdown import Markdown
import yaml
from dataclasses import dataclass
import asyncio
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("moe_v4.log"), logging.StreamHandler()])

# Initialize Console for rich output
console = Console()

# Load configuration from YAML file
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("Error: config.yaml not found. Please create a config.yaml file.")
    exit(1)
except yaml.YAMLError as e:
    logging.error(f"Error parsing config.yaml: {e}")
    exit(1)

# API Keys - Load from environment variables or Google Colab userdata
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

try:
    from google.colab import userdata
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")
    if not XAI_API_KEY:
        XAI_API_KEY = userdata.get("XAI_API_KEY")
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
except ImportError:
    logging.warning("google.colab module not found. Using environment variables for API keys.")

# Data class to hold the results
@dataclass
class WorkflowResults:
    """Data class to hold the results of the workflow."""
    OpenAI: str = ""
    Anthropic: str = ""
    xAI: str = ""
    Consensus_Analysis: str = ""
    Charts_Mindmaps: str = ""
    Analysis_Tools: str = ""
    Related_Questions: str = ""
    Meta_Analysis: str = ""

# Define expert styles from config
expert_styles = config.get("expert_styles", {})

def create_llm_model(model_name: str, model_config: Dict) -> Any:
    """
    Creates an LLM model instance based on the provided model name and configuration.

    Args:
        model_name (str): The name of the LLM model to use (e.g., "openai", "anthropic", "xai", "google").
        model_config (Dict): A dictionary containing the configuration for the LLM model.

    Returns:
        Any: An instance of the LLM model.
    """
    try:
        if model_name == "openai":
            return ChatOpenAI(model=model_config.get("model", config["openai_model"]),
                             temperature=model_config.get("temperature", 0),
                             max_tokens=model_config.get("max_tokens", 256),
                             api_key=OPENAI_API_KEY)
        elif model_name == "anthropic":
            return ChatAnthropic(model=model_config.get("model", config["anthropic_model"]),
                                temperature=model_config.get("temperature", 0),
                                max_tokens=model_config.get("max_tokens", 256),
                                api_key=ANTHROPIC_API_KEY)
        elif model_name == "xai":
            return ChatXAI(model=model_config.get("model", config["xai_model"]),
                          temperature=model_config.get("temperature", 0),
                          max_tokens=model_config.get("max_tokens", 256),
                          api_key=XAI_API_KEY)
        elif model_name == "google":
            return ChatGoogleGenerativeAI(model=model_config.get("model", config["supervisor_model"]),
                                         temperature=model_config.get("temperature", 0),
                                         max_tokens=model_config.get("max_tokens", 512),
                                         api_key=GOOGLE_API_KEY)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    except Exception as e:
        logging.error(f"Error creating LLM model {model_name}: {e}")
        return None

def create_expert(model_name: str, style: str, model_config: Dict) -> Callable[[str], str]:
    """
    Creates an expert LLM with a specific style.

    Args:
        model_name (str): The name of the LLM model to use (e.g., "openai", "anthropic", "xai").
        style (str): The style of the expert (e.g., "technical", "creative", "business").
        model_config (Dict): A dictionary containing the configuration for the LLM model.

    Returns:
        Callable[[str], str]: A function that takes a query and returns the LLM's response.
    """
    style_prompt = f"You are an expert with style: {style}."
    model = create_llm_model(model_name, model_config)
    if not model:
        return lambda query: f"Error: Could not create expert {model_name}"

    async def invoke_expert(query: str) -> str:
        """Invokes the expert LLM with a given query."""
        try:
            response = await model.ainvoke([("system", style_prompt), ("user", query)])
            return response.content
        except Exception as e:
            logging.error(f"Error invoking expert {model_name}: {e}")
            return f"Error: Could not invoke expert {model_name}"
    return invoke_expert

# Create experts with different styles
openai_expert = create_expert("openai", "technical", config.get("openai_config", {}))
anthropic_expert = create_expert("anthropic", "creative", config.get("anthropic_config", {}))
xai_expert = create_expert("xai", "business", config.get("xai_config", {}))

# Initialize supervisor model
supervisor_model = create_llm_model("google", config.get("supervisor_config", {}))

async def invoke_llm(model: Any, role: str, content: str, task: str) -> str:
    """
    Invokes an LLM with a system prompt.

    Args:
        model (Any): The LLM model to use.
        role (str): The role of the LLM (e.g., "analyzing responses", "generating charts").
        content (str): The content to provide to the LLM.
        task (str): The task to instruct the LLM to perform.

    Returns:
        str: The response from the LLM.
    """
    logging.info(f"Invoking LLM as {role} to {task}")
    prompt = [
        ("system", config["prompts"].get(f"{role}_system", f"You are a supervisor {role}. {task}")),
        ("user", content)
    ]
    try:
        response = await model.ainvoke(prompt)
        return response.content
    except Exception as e:
        logging.error(f"Error invoking LLM: {e}")
        return f"Error: Could not invoke LLM for {task}"

async def get_expert_responses(query: str) -> Dict[str, str]:
    """
    Gathers responses from different expert LLMs asynchronously.

    Args:
        query (str): The query to send to the expert LLMs.

    Returns:
        Dict[str, str]: A dictionary containing the responses from each expert.
    """
    logging.info("Gathering insights from our AI experts...")
    tasks = [
        openai_expert(query),
        anthropic_expert(query),
        xai_expert(query)
    ]
    responses = await asyncio.gather(*tasks)
    return {
        "OpenAI": responses[0],
        "Anthropic": responses[1],
        "xAI": responses[2]
    }

async def analyze_responses(responses: Dict[str, str], analysis_type: str) -> str:
    """
    Analyzes the responses using a specific analysis type asynchronously.

    Args:
        responses (Dict[str, str]): A dictionary containing the responses from each expert.
        analysis_type (str): The type of analysis to perform (e.g., "consensus", "charts", "tools", "questions", "meta").

    Returns:
        str: The analysis result from the supervisor LLM.
    """
    logging.info(f"Performing {analysis_type} analysis...")
    task = config["prompts"].get(f"{analysis_type}_task", f"Perform {analysis_type} analysis.")
    content = "\n".join([f"{name}: {resp}" for name, resp in responses.items()]) if analysis_type == "consensus" else f"Content:\n\n{responses}"
    role = f"analyzing {analysis_type}"
    return await invoke_llm(supervisor_model, role, content, task)

async def run_full_workflow(query: str) -> WorkflowResults:
    """
    Runs the full analysis workflow asynchronously.

    Args:
        query (str): The query to analyze.

    Returns:
        WorkflowResults: A dataclass containing the results of the workflow.
    """
    logging.info("Initiating the full analysis workflow...")
    responses = await get_expert_responses(query)
    combined_responses = "\n".join([f"{name}:\n{resp}" for name, resp in responses.items()])

    results = WorkflowResults(
        OpenAI=responses.get("OpenAI", ""),
        Anthropic=responses.get("Anthropic", ""),
        xAI=responses.get("xAI", ""),
        Consensus_Analysis=await analyze_responses(responses, "consensus"),
        Charts_Mindmaps=await analyze_responses(combined_responses, "charts"),
        Analysis_Tools=await analyze_responses(combined_responses, "tools"),
        Related_Questions=await analyze_responses(combined_responses, "questions"),
        Meta_Analysis=await analyze_responses(combined_responses, "meta")
    )
    return results

def display_results(results: WorkflowResults, query_example: str) -> None:
    """
    Displays the results using the rich library.

    Args:
        results (WorkflowResults): The results of the workflow.
        query_example (str): The original query.
    """
    console.print("[bold blue]=== Workflow Results ===[/bold blue]")
    console.print(f"[italic]Query:[/italic] {query_example}\n")

    console.print("[bold green]=== Expert Responses ===[/bold green]")
    console.print(f"[bold]OpenAI:[/bold]\n{results.OpenAI}\n")
    console.print(f"[bold]Anthropic:[/bold]\n{results.Anthropic}\n")
    console.print(f"[bold]xAI:[/bold]\n{results.xAI}\n")

    if results.Consensus_Analysis:
        console.print("[bold magenta]=== Consensus Analysis ===[/bold magenta]")
        console.print(Markdown(results.Consensus_Analysis))
        console.print("\n")

    sections = {
        "Charts_Mindmaps": "Charts and Mindmaps",
        "Analysis_Tools": "Analysis Tools",
        "Related_Questions": "Related Questions",
        "Meta_Analysis": "Meta Analysis"
    }

    for key, title in sections.items():
        if getattr(results, key, None):
            console.print(f"[bold yellow]=== {title} ===[/bold yellow]")
            console.print("\n")
            console.print(Markdown(getattr(results, key)))
            console.print("\n")
            console.print("\n")
            console.print("\n")

async def main():
    query_example = "Explain how Data analysis and data science are different"
    results = await run_full_workflow(query_example)
    display_results(results, query_example)

if __name__ == "__main__":
    asyncio.run(main())