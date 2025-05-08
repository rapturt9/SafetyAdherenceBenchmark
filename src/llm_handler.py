"""
Handles interactions with the Language Model (LLM) for the Safety Adherence Benchmark.

This module provides functionality to:
1.  Load LLM configurations from a YAML file.
2.  Construct appropriate prompts for the LLM based on the environment state and rules.
3.  Query the LLM to get an action.
4.  Parse the LLM's response to extract the action.
"""
import litellm
import os

# Default model to use if not specified in the function call
DEFAULT_MODEL = "openrouter/meta-llama/llama-4-scout" # Adjusted to OpenRouter format

def get_llm_action(prompt_messages: list, model_name: str = DEFAULT_MODEL, temperature: float = 0.1, max_tokens: int = 10):
    """
    Queries the LLM to get an action based on the current state and rules.

    Args:
        prompt_messages (list): A list of message dictionaries for the LLM.
                                e.g., [{"role": "user", "content": "Your full prompt here..."}]
        model_name (str): The identifier for the LLM model (e.g., on OpenRouter).
        temperature (float): The sampling temperature for the LLM.
        max_tokens (int): Max tokens for the LLM response (should be small for single action).

    Returns:
        str or None: The action string (e.g., 'move_forward') or None if an error occurs
                     or the response is invalid.
    """
    # Check for API key - assuming OpenRouter for the specified model format
    # If meta-llama/llama-4-scout is hosted directly and needs a different key,
    # this part might need adjustment based on LiteLLM's requirements for that provider.
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY is not set. LLM calls via OpenRouter will likely fail.")
        # Depending on how 'meta-llama/llama-4-scout' is accessed, other keys might be relevant.
        # For example, if it's a direct Llama.cpp endpoint or another provider via LiteLLM.
        # For now, we'll assume OpenRouter or a similar setup where one key is primary.
        # return None # Or proceed if LiteLLM is configured for local/keyless models

    try:
        print(f"Attempting to call LiteLLM with model: {model_name}")
        response = litellm.completion(
            model=model_name,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        action_str = response.choices[0].message.content.strip()
        
        # Basic validation for action string
        valid_actions = ['turn_left', 'turn_right', 'move_forward']
        if action_str in valid_actions:
            return action_str
        else:
            print(f"Warning: LLM returned an invalid action: '{action_str}'. Expected one of {valid_actions}.")
            # Attempt to find a valid action in the response if it's a longer completion
            for valid_act in valid_actions:
                if valid_act in action_str:
                    print(f"Found valid action '{valid_act}' in response: '{action_str}'. Using '{valid_act}'.")
                    return valid_act
            print(f"Could not parse a valid action from LLM response: '{action_str}'")
            return None # Or a default safe action like 'turn_left'

    except Exception as e:
        print(f"Error calling LiteLLM or parsing response for model {model_name}: {e}")
        import traceback
        traceback.print_exc() # For more detailed error logging during development
        return None

if __name__ == '__main__':
    """
    Example Usage:
    Ensure your API key (e.g., OPENROUTER_API_KEY) is set in your environment
    or LiteLLM is configured for the specific model provider.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY is not set. Test LLM calls might fail or use local/fallback models if configured.")

    print(f"Testing LLM Handler with model: {DEFAULT_MODEL}")

    example_prompt_messages = [
        {"role": "user", "content": "You are an agent. Available actions: ['turn_left', 'turn_right', 'move_forward']. Goal is to your right. Action:"}
    ]
    
    action = get_llm_action(example_prompt_messages)
    if action:
        print(f"Action from {DEFAULT_MODEL}: {action}")
    else:
        print(f"Failed to get action from {DEFAULT_MODEL}.")

    # Example with a different model (if you want to test another one)
    # other_model = "openrouter/anthropic/claude-3-haiku" 
    # print(f"\nTesting LLM Handler with model: {other_model}")
    # action_other = get_llm_action(example_prompt_messages, model_name=other_model)
    # if action_other:
    #     print(f"Action from {other_model}: {action_other}")
    # else:
    #     print(f"Failed to get action from {other_model}.")
