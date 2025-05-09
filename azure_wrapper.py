# azure_openai_wrapper.py

import os
from openai import AsyncAzureOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled
from dotenv import load_dotenv

def init_azure_openai(api_version="2023-12-01-preview", default_deployment="gpt-4"):
    """
    Initializes and returns an Azure OpenAI client.
    Also sets it as default client for the OpenAI Agents SDK.
    """
    set_tracing_disabled(True)
    load_dotenv()

    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    # Clean up deployment name in case it includes comment markers
    raw_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", default_deployment)
    azure_deployment = raw_deployment.split("#")[0].strip()
    
    print(f"Azure OpenAI Configuration:")
    print(f"- Endpoint: {azure_endpoint}")
    print(f"- Deployment: {azure_deployment}")
    print(f"- API Version: {api_version}")

    if not azure_api_key or not azure_endpoint:
        raise ValueError("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in environment variables")

    azure_client = AsyncAzureOpenAI(
        api_key=azure_api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )

    set_default_openai_client(client=azure_client, use_for_tracing=False)
    set_default_openai_api("chat_completions")

    return azure_client, azure_deployment