import streamlit as st
import json
import re
from typing import List, TypedDict, Optional

import requests
import certifi
from bs4 import BeautifulSoup
import html2text

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- 1. CORE LOGIC & HELPER FUNCTIONS ---

# --- Data Processing (from your new script) ---

def process_uploaded_json(uploaded_file):
    """
    Processes the uploaded JSON file to extract task details, convert HTML to Markdown,
    and find all URLs within the content.
    """
    st.write("Processing uploaded JSON file...")
    try:
        # Instantiate html2text converter
        h = html2text.HTML2Text()
        h.ignore_links = False  # Keep links in the markdown

        # Load JSON data
        data = json.load(uploaded_file)

        # Validate JSON structure
        if not ('data' in data and data['data'] and isinstance(data['data'], list)):
            st.error("JSON format error: Expected a 'data' key with a list of items.")
            return None, None

        task_data = data['data'][0]
        name = task_data.get('name', '')
        description = task_data.get('description', '')
        instructions = task_data.get('instructions', '')

        # Convert HTML content to Markdown
        name_md = h.handle(name)
        description_md = h.handle(description)
        instructions_md = h.handle(instructions)

        # Combine into a single markdown document
        full_task_md = f"# Name\n{name_md}\n\n## Description\n{description_md}\n\n## Instructions\n{instructions_md}"
        st.write("Successfully converted HTML to Markdown.")

        # Extract all URLs from the final markdown text
        # Regex for markdown links and plain URLs
        url_pattern = re.compile(r'https?://[^\s<>"\'\]\),]+')
        urls = url_pattern.findall(full_task_md)
        unique_urls = list(set(urls))

        # Filter out non-scrapeable files like PDFs
        filtered_urls = [url for url in unique_urls if not url.lower().endswith(('.pdf', '.zip', '.jpg'))]
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

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="Task Update Assistant", layout="wide")
st.title("Task Update Tool")
st.markdown("This tool analyzes tasks from a JSON file, scrapes content from URLs linked within it, and suggests updates using an LLM")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    provider = st.selectbox("Select LLM Provider", ["Google"])

    api_key = None
    if provider == "Google":
        api_key = st.text_input("Google API Key", type="password")
        st.markdown("Get key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
    else: 
        None
    uploaded_file = st.file_uploader(
        "Upload Task File in JSON format",
        type=['json'],
        help="The JSON should have a `data` key containing a list with one object, which has `name`, `description`, and `instructions` fields (can be HTML)."
    )

# --- Main Page Logic ---
if st.button("Propose Updates", disabled=(not uploaded_file or not api_key)):
    
    full_task_md, task_urls = process_uploaded_json(uploaded_file)
    
    if full_task_md and task_urls is not None:
        
        with st.spinner("Initializing workflow and running analysis... This may take a few minutes."):
            try:
                # --- 3. DYNAMICALLY BUILD AND RUN THE GRAPH ---
                
                # Initialize the correct LLM based on user choice
                if provider == "Google":
                    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7, google_api_key=api_key)
                else: # OpenAI
                    llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0)

                # Define the graph
                workflow = StateGraph(GraphState)

                # Add nodes
                workflow.add_node("scrape_urls", scrape_urls)
                workflow.add_node("propose_update", lambda state: propose_update(state, llm))

                # Define the flow - Simplified based on new logic
                workflow.set_entry_point("scrape_urls")
                workflow.add_edge("scrape_urls", "propose_update")
                workflow.add_edge("propose_update", END)

                # Compile and run the graph
                app = workflow.compile()
                initial_state = {"task": full_task_md, "task_urls": task_urls}
                result = app.invoke(initial_state)

                # --- 4. DISPLAY RESULTS ---
                st.success("Analysis complete!")
                st.header("Proposed Updates")

                # Display the processed markdown task for reference
                with st.expander("View Full Task Content (as processed)"):
                    st.markdown(full_task_md)

                proposed_updates = result.get("proposed_update", {})

                if not proposed_updates:
                    st.info("No updates were proposed.")
                else:
                    for url, update_data in proposed_updates.items():
                        with st.expander(f"**Source: {url}**", expanded=True):
                            if "error" in update_data:
                                st.error(f"An error occurred: {update_data['error']}")
                                continue
                            
                            discrepancies = update_data.get('discrepancies', [])
                            improvements = update_data.get('general_improvements', [])

                            if not discrepancies and not improvements:
                                st.success("No discrepancies found. The instructions appear to be up-to-date based on this source.")
                                continue

                            if discrepancies:
                                st.subheader("Discrepancies Found")
                                for item in discrepancies:
                                    st.markdown(f"**Reasoning:** {item.get('description_reasoning')}")
                                    st.markdown(f"**Severity:** `{item.get('severity', 'N/A').upper()}`")
                                    update = item.get('suggested_update', {})
                                    if update.get('original_text') or update.get('revised_text'):
                                        st.markdown("**Suggested Change:**")
                                        col1, col2 = st.columns(2)
                                        col1.text_area("Original Text", value=update.get('original_text', 'N/A'), height=150, disabled=True)
                                        col2.text_area("Revised Text", value=update.get('revised_text', 'N/A'), height=150, disabled=True)
                                    st.divider()
                            
                            if improvements:
                                st.subheader("General Improvements")
                                for item in improvements:
                                    st.markdown(f"**Suggestion:** {item.get('description_reasoning')}")
                                    update = item.get('suggested_update', {})
                                    if update.get('original_text') or update.get('revised_text'):
                                        col1, col2 = st.columns(2)
                                        col1.text_area("Original Section", value=update.get('original_text', 'N/A'), height=150, disabled=True, key=f"imp_orig_{url}")
                                        col2.text_area("Suggested Improvement", value=update.get('revised_text', 'N/A'), height=150, disabled=True, key=f"imp_rev_{url}")
                                    st.divider()
                                    
            except Exception as e:
                st.error(f"An unexpected error occurred during the workflow: {e}")
                st.exception(e)

elif not (uploaded_file and api_key):
    st.warning("Please obtain and provide API Key and upload a task JSON file")
