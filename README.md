# Multi-Agent Orchestration (MOE) System v4

## Overview

This Python script, `moe_v4.py`, implements a Multi-Agent Orchestration (MOE) system using various Large Language Models (LLMs) to analyze a given query. It follows a structured workflow, gathering insights from different LLMs with specific styles, analyzing consensus, generating visual representations, performing detailed analysis, suggesting related questions, and conducting meta-analysis. The script is designed to be highly configurable, robust, and easy to maintain, with asynchronous operations for improved performance.

## Key Features

*   **Asynchronous Operations:** Uses `asyncio` for concurrent execution, improving performance.
*   **Structured Logging:** Logs are written to both console and a file (`moe_v4.log`).
*   **Granular Configuration:** Allows more granular control over LLM parameters (temperature, max tokens) via `config.yaml`.
*   **Modular Design:** The code is organized into well-defined functions, each responsible for a specific task.
*   **Configurability:** Parameters such as model names, API keys, prompts, and styles are loaded from a YAML configuration file (`config.yaml`).
*   **Robustness:** Includes error handling and logging to ensure smooth execution and easier debugging.
*   **Type Hinting:** Uses type hints for better code readability and maintainability.
*   **Comprehensive Documentation:** Includes docstrings for all functions and detailed comments throughout the code.
*   **Enhanced Output:** Uses the `rich` library for visually appealing and informative output.
*   **API Key Management:** Uses `.env` files and `python-dotenv` for secure API key management.
*   **Externalized Prompts:** Prompts are now loaded from the `config.yaml` file.
*   **Flexible Analysis:** Uses a single `analyze_responses` function for all analysis types.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install langchain-openai langchain-anthropic langchain-xai langchain-google-genai python-dotenv rich pyyaml
    ```

2.  **Create `.env` File:**

    Create a `.env` file in the same directory as `moe_v4.py` and add your API keys:

    ```
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    XAI_API_KEY=your_xai_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

3.  **Create `config.yaml` File:**

    Create a `config.yaml` file in the same directory as `moe_v4.py` and add your configuration:

    ```yaml
    openai_model: "gpt-4o"
    anthropic_model: "claude-3-5-haiku-20241022"
    xai_model: "grok-beta"
    supervisor_model: "gemini-2.0-flash-exp"

    openai_config:
      temperature: 0.1
      max_tokens: 300

    anthropic_config:
      temperature: 0.2
      max_tokens: 350

    xai_config:
      temperature: 0.0
      max_tokens: 280

    supervisor_config:
      temperature: 0.0
      max_tokens: 512

    expert_styles:
      technical: "Focus on detailed technical explanations."
      creative: "Use imaginative, broad storytelling approaches."
      business: "Emphasize strategic and economic impacts."

    prompts:
      consensus_task: "Analyze the following experts' responses. Provide a consensus analysis and highlight disagreements."
      charts_task: "Generate useful charts or mindmap descriptions in concise text."
      tools_task: "Perform sentiment analysis, bias detection, uncertainty highlighting, and jargon explanation. Separate each analysis by sections."
      questions_task: "Provide related questions for deeper learning."
      meta_task: "Evaluate quality metrics and perform pattern recognition."
    ```

4.  **Run the Script:**

    ```bash
    python moe_v4.py
    ```

## Usage

1.  Modify the `config.yaml` file to customize the LLM models, styles, prompts, and LLM parameters.
2.  Set your API keys in the `.env` file.
3.  Run the `moe_v4.py` script.
4.  Check the `moe_v4.log` file for detailed logs.

## Code Overview

*   **`moe_v4.py`:** The main script containing the implementation of the MOE system.
*   **`config.yaml`:** Configuration file for LLM models, styles, prompts, and LLM parameters.
*   **`.env`:** File for storing API keys.
*   **`moe_v4.log`:** Log file for detailed execution logs.
*   **`README.md`:** This file, providing an overview of the project and usage instructions.

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues.

## License

This project is licensed under the MIT License.
