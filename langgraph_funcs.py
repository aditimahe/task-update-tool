# --- 1. CORE LOGIC & HELPER FUNCTIONS ---
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, TypedDict, Optional
import streamlit as st
import json
import re

import requests
import certifi
from bs4 import BeautifulSoup
# import html2text

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# --- Data Processing (from your new script) ---

def html_to_text(html):
    """Converts HTML to plain text, formatting links as 'text (URL)'."""
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            a.replace_with(f"{a.get_text()} ({href})")
        else:
            a.unwrap()
    return soup.get_text()

def obtain_full_task_md(data):
    """Combines task name, description, and instructions into a single text block."""
    st.write('→ Obtaining full task in plain text format...')
    name_md = html_to_text(data['data'][0]['name'])
    description_md = html_to_text(data['data'][0]['description'])
    instructions_md = html_to_text(data['data'][0]['instructions'])
    full_task_md = f"# Name\n{name_md}\n\n## Description\n{description_md}\n\n## Instructions\n{instructions_md}"
    return full_task_md

def obtain_urls_from_existing_task(full_task):
    """Extracts and filters URLs from the task text."""
    st.write('→ Obtaining URLs from existing task...')
    # Regex to find URLs inside parentheses, which matches the output of html_to_text
    urls = re.findall(r'\((https?://[^\s)]+)\)', full_task)

    # Remove duplicates
    unique_urls = list(set(urls))

    # Filter out URLs ending with .pdf (case-insensitive)
    filtered_urls = [url for url in unique_urls if not url.lower().endswith('.pdf')]

    return filtered_urls

def get_soups(urls):
    """Fetches web pages and converts their HTML to plain text."""
    soups_md = {}
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup_md = html_to_text(response.text)
            soups_md[url] = soup_md
            st.write(f"Fetched and parsed: {url}")
        except requests.RequestException as e:
            st.warning(f"Failed to fetch {url}: {e}")
    return soups_md

# --- Data Processing (Modified to call the new helper functions) ---

def process_uploaded_json(uploaded_file):
    """
    Processes the uploaded JSON file by calling helper functions to extract task details,
    convert HTML to plain text, and find all URLs.
    """
    st.write("Processing uploaded JSON file...")
    try:
        # Load JSON data
        data = json.load(uploaded_file)

        # Validate JSON structure
        if not ('data' in data and data['data'] and isinstance(data['data'], list)):
            st.error("JSON format error: Expected a 'data' key with a list of items.")
            return None, None

        # Call the helper function to convert HTML and structure the text
        full_task_md = obtain_full_task_md(data)
        st.write("Successfully converted HTML to plain text.")

        # Call the helper function to extract URLs from the generated text
        filtered_urls = obtain_urls_from_existing_task(full_task_md)
        st.write(f"Found {len(filtered_urls)} relevant URLs to analyze.")

        return full_task_md, filtered_urls

    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a correctly formatted file.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        return None, None

# --- LangGraph Pydantic Models and State ---

class TaskUpdateResponse(BaseModel):
    """The final structured response for task updates."""
    discrepancies: List[dict] = Field(default_factory=list)
    general_improvements: Optional[List[dict]] = Field(default_factory=list)

class GraphState(TypedDict):
    """Defines the state for our graph."""
    task: str
    task_urls: List[str]
    documents: dict
    proposed_update: dict

# --- LangGraph Node Functions ---

def scrape_urls(state: GraphState) -> GraphState:
    """Scrapes the content from the provided URLs."""
    urls_to_scrape = state.get("task_urls", [])
    if not urls_to_scrape:
        st.warning("No URLs found to scrape.")
        return {**state, "documents": {}}

    st.write(f"Scraping content from {len(urls_to_scrape)} URL(s)...")
    documents = {}
    progress_bar = st.progress(0, text="Scraping progress...")

    for i, url in enumerate(urls_to_scrape):
        try:
            with st.spinner(f"Scraping {url}..."):
                response = requests.get(url, verify=certifi.where(), timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "lxml")
                
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                text = " ".join(p.get_text(strip=True) for p in main_content.find_all("p"))
                
                if text:
                    documents[url] = text
                    st.write(f"Successfully scraped {url}")
                else:
                    st.write(f"No paragraph text found at {url}")

        except requests.RequestException as e:
            st.error(f"Error scraping {url}: {e}")
            continue
        finally:
            progress_bar.progress((i + 1) / len(urls_to_scrape))
    
    progress_bar.empty()
    return {**state, "documents": documents}

def propose_update(state: GraphState, llm) -> GraphState:
    """Analyzes content and proposes updates using the provided LLM."""
    task = state["task"]
    documents = state["documents"]
    
    if not documents:
        st.warning("No documents were scraped. Skipping update proposal.")
        return {**state, "proposed_update": {}}

    st.write("Generating update proposals using the LLM...")
    
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert analyst. Your task is to analyze website content to determine if a set of instructions needs updating.

        Your goal is to:
        1. Identify discrepancies or outdated information in the instructions based on the provided website content.
        2. Suggest specific, actionable updates, focusing only on the content that requires revision.
        3. Clearly explain the reasoning behind each suggested change.
        4. If the instructions are already up-to-date, state that no changes are needed.

        **Important: Your response must be a valid JSON object matching this schema:**

        {{
          "discrepancies": [
            {{
              "description_reasoning": "string",
              "severity": "minor|major",
              "suggested_update": {{
                "original_text": "string or null",
                "revised_text": "string or null"
              }}
            }}
          ],
          "general_improvements": [
            {{
              "description_reasoning": "string",
              "suggested_update": {{
                "original_text": "string or null",
                "revised_text": "string or null"
              }}
            }}
          ]
        }}

        If no updates are needed, return a JSON with empty lists for "discrepancies" and "general_improvements".

        ---
        Original Instructions:
        {task}
        ---
        Context from Website ({url}):
        {context}
        ---

        Your response in valid JSON:
        """
    )

    update_chain = prompt_template | llm | JsonOutputParser(pydantic_object=TaskUpdateResponse)

    updates = {}
    progress_bar = st.progress(0, text="LLM analysis progress...")
    
    doc_items = list(documents.items())
    for i, (url, doc_content) in enumerate(doc_items):
        with st.spinner(f"Analyzing content from {url}..."):
            try:
                # Truncate content to avoid exceeding token limits
                truncated_content = doc_content[:15000]
                
                result = update_chain.invoke({
                    "task": task, 
                    "context": truncated_content,
                    "url": url
                })
                updates[url] = result
                st.write(f"Analysis complete for {url}")

            except Exception as e:
                st.error(f"Error analyzing content from {url}: {e}")
                updates[url] = {"error": f"Failed to process this URL. Details: {str(e)}"}
        progress_bar.progress((i + 1) / len(doc_items))

    progress_bar.empty()
    return {**state, "proposed_update": updates}