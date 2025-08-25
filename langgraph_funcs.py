import re
import requests
import certifi
import time
from bs4 import BeautifulSoup
import streamlit as st

from typing import List, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field, ValidationError

from duckduckgo_search import DDGS
from datetime import datetime 


# LangGraph imports
from langgraph.graph import StateGraph, END

class StatusTracker:
    """A helper class to manage and update the Streamlit UI for progress tracking."""
    def __init__(self):
        self.url_statuses = {}
        self.placeholders = {}
        self.container = st.expander("📊 Live Analysis Dashboard", expanded=True)

    def add_urls(self, urls, url_type):
        """Add new URLs to the tracker."""
        for url in urls:
            if url not in self.url_statuses:
                self.url_statuses[url] = {"type": url_type, "status": "Pending", "icon": "⏳"}
                with self.container:
                    self.placeholders[url] = st.empty()
                self.update_url_display(url)

    def update_status(self, url, status, icon):
        """Update the status of a specific URL."""
        if url in self.url_statuses:
            self.url_statuses[url]["status"] = status
            self.url_statuses[url]["icon"] = icon
            self.update_url_display(url)

    def update_url_display(self, url):
        """Refresh the Streamlit placeholder for a specific URL."""
        if url in self.placeholders:
            info = self.url_statuses[url]
            badge_color = "blue" if info['type'] == 'Direct' else "orange"
            # Truncate long URLs for display
            display_url = (url[:70] + '...') if len(url) > 73 else url
            markdown_string = (
                f"{info['icon']} **{info['status']}** "
                f"<span style='color:{badge_color}; border: 1px solid {badge_color}; border-radius: 5px; padding: 2px 6px; font-size: 0.8em;'>{info['type']}</span>"
                f"<br><small style='color:grey;'>{display_url}</small>"
            )
            self.placeholders[url].markdown(markdown_string, unsafe_allow_html=True)

    def update_from_graph_state(self, current_node, state):
        """Intelligently update statuses based on the current graph state."""
        if current_node == "extract_task_urls":
            self.add_urls(state.get("task_urls", []), "Direct")

        elif current_node == "search_duckduckgo":
            self.add_urls(state.get("search_urls", []), "Discovered")

        elif current_node == "scrape_urls":
            scraped_urls = list(state.get("relevant_documents", {}).keys()) + list(state.get("indirect_documents", {}).keys())
            for url in self.url_statuses:
                if url in scraped_urls:
                    self.update_status(url, "Scraped, Awaiting Grade", "🤔")
                elif self.url_statuses[url]['status'] == 'Pending': # If it wasn't scraped, it might be in progress
                    self.update_status(url, "Scraping", "📄")


        elif current_node == "grade_indirect_content":
            relevant = state.get("relevant_documents", {})
            indirect = state.get("indirect_documents", {})
            for url, info in self.url_statuses.items():
                if info['type'] == 'Discovered':
                    if url in relevant:
                        self.update_status(url, "Graded as RELEVANT", "✅")
                    elif url in indirect: # If it's still in indirect, it was not promoted to relevant
                        self.update_status(url, "Graded as NOT RELEVANT", "❌")
                elif info['type'] == 'Direct': # Direct sources are assumed relevant
                    self.update_status(url, "Source is RELEVANT", "✅")


        elif current_node == "propose_update":
            analyzed_urls = state.get("proposed_update", {}).keys()
            for url in analyzed_urls:
                 self.update_status(url, "Analysis Complete", "🎉")

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Replace <a> tags with text (text + URL)
    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            a.replace_with(f"{a.get_text()} ({href})")
        else:
            a.unwrap()

    return soup.get_text()

def obtain_full_task_md(data):
    print('→ Obtaining full task in Markdown format...')
    name_md = html_to_text(data['data'][0]['name'])
    description_md = html_to_text(data['data'][0]['description'])
    instructions_md = html_to_text(data['data'][0]['instructions'])
    full_task_md = f"# Name\n{name_md}\n\n## Description\n{description_md}\n\n## Instructions\n{instructions_md}"
    return full_task_md

def extract_task_urls(state):
    print("---EXTRACTING TASK URLS---")
    task_urls = []
    direct_urls = []
    additional_task_urls = []

    task = state["task"]
    url_pattern = re.compile(
        r'https?://[^\s<>"\'\]\),]+'
    )
    direct_urls = url_pattern.findall(task)
    task_urls = [*direct_urls]
    print('Direct URLs:')
    for url in list(set(direct_urls)):
        print(url)

    if state['run_settings']['additional_sources']:
        additional_task_urls = state['run_settings']['additional_sources']
        task_urls = [*task_urls, *additional_task_urls]
        print('Additional URLs:')
        for url in additional_task_urls:
            print(url)

    print('='*50)
    print(f'Found {len(task_urls)} urls, {len(direct_urls)} direct, {len(additional_task_urls)} additional')
    print('='*50)

    return {**state, "task_urls": list(set(task_urls))}

def scrape_urls_func(all_urls, type_to_save):
    documents = {}

    for url in all_urls:
        if url.lower().endswith(".pdf"):
            print(f"Skipping PDF URL: {url}")
            continue

        print('\n→ Scraping URL:', url)
        try:
            response = requests.get(url, verify=certifi.where(), timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            documents[url] = {
                "type": type_to_save,
                "text": text,
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    return documents

def scrape_urls(state):
    print("---SCRAPING URLS---")
    task_urls = state.get("task_urls", [])
    search_urls = state.get("search_urls", [])
    all_urls = list(set(task_urls + search_urls))
    if state['run_settings']['direct_sources']:
        documents_relevant = scrape_urls_func(task_urls, "direct")
        indirect_documents = scrape_urls_func(search_urls, "indirect")
    else: 
        documents_relevant = {}
        indirect_documents = {}

    return {**state, "urls": all_urls, "indirect_documents": indirect_documents, "relevant_documents": documents_relevant}

# prompt = ChatPromptTemplate.from_template(
#     """
#     You are tasked with analyzing the website content to determine whether the provided instructions need to be updated.

#     Your goal is to:
#     1. Identify any discrepancies or outdated information in the instructions based on the website content.
#     2. Assess the severity of each discrepancy — classify it as either minor or major.
#     3. Suggest specific, actionable updates to the instructions, focusing only on the content that requires revision.
#     4. Clearly explain the reasoning behind each suggested change.
#     5. If applicable, propose general improvements to enhance the clarity, completeness, or level of detail in the instructions.

#     Only evaluate and suggest updates related to the content of the instructions — do not rewrite or reformat unrelated parts found in the website content.

#     **Important: Your response must be valid JSON matching this schema:**

#     {{
#       "discrepancies": [
#         {{
#           "description_reasoning": "string",
#           "severity": "minor|major",
#           "suggested_update": {{
#             "original_text": "string or null",
#             "revised_text": "string or null",
#           }}
#         }}
#       ],
#       "general_improvements": [
#         {{
#           "description_reasoning": "string",
#           "suggested_update": {{
#             "original_text": "string or null",
#             "revised_text": "string or null",
#           }}
#         }}
#       ]
#     }}

#     Original Task:
#     {task}

#     Context from current sources:
#     {context}

#     Your response in valid json:
#     """
# )

class SuggestedUpdate(BaseModel):
    original_text: Optional[str] = Field(None, description="The original instruction text to update")
    revised_text: Optional[str] = Field(None, description="The suggested revision")

class Discrepancy(BaseModel):
    description_reasoning: str = Field(..., description="Description of the discrepancy and reasoning")
    severity: str = Field(..., description="Severity level: 'minor' or 'major'")
    suggested_update: SuggestedUpdate

class GeneralImprovement(BaseModel):
    description_reasoning: str = Field(..., description="Description and reasoning for improvement")
    suggested_update: SuggestedUpdate

class TaskUpdateResponse(BaseModel):
    discrepancies: List[Discrepancy] = Field(default_factory=list)
    general_improvements: Optional[List[GeneralImprovement]] = Field(default_factory=list)

def propose_update(state):
    print("---PROPOSING UPDATES BASED ON SCRAPED CONTENT---")
    task = state["task"]
    llm = state["llm"] # Get llm from state
    updates = {}

    user_prompt_template = state["update_prompt"]
    prompt = ChatPromptTemplate.from_template(user_prompt_template)
    update_chain = prompt | llm | StrOutputParser()

    print("Relevant documents for analysis:", list(state["relevant_documents"].keys()))

    if not state["relevant_documents"]:
        print("No relevant documents found to propose updates.")
        return {**state, "proposed_update": {}}

    for url, doc in state["relevant_documents"].items():
        raw_result = update_chain.invoke({"task": task, "context": doc['text']})
        print('='*100)
        print('~~~~~ URL:', url)
        print('RAW RESULT:\n', raw_result)
        print('='*100)

        try:
            # parsed_result = json.loads(raw_result)
            json_parser = JsonOutputParser()
            parsed_result = json_parser.parse(raw_result)
            # Optionally validate with Pydantic:
            validated_result = TaskUpdateResponse.model_validate(parsed_result)
            updates[url] = {
                "type": "direct",
                "update": validated_result.model_dump()
            }

        except json.JSONDecodeError as e:
            # Handle invalid JSON response gracefully
            print("Error: LLM output was not valid JSON:", e)
            print("Raw output:", raw_result)
            updates[url] = 'Invalid JSON from LLM'
        except ValidationError as e:
            print("Error: Output JSON did not match schema:", e)
            updates[url] = 'Schema validation failed'

    return {**state, "proposed_update": updates}

def should_search(state):
    print("---DECISION: SHOULD WE SEARCH?---")
    st.write("---DECISION: SHOULD WE SEARCH?---")
    if state.get("run_settings", {})['indirect_sources']:
        print("Decision: Proceeding to search the web for external sources.")
        return "generate_search_terms"
    else:
        print("Decision: Not searching the web for external sources.")
        return "scrape_urls"

def search_duckduckgo(state):
    print("---SEARCHING THE WEB---")
    st.write("---SEARCHING THE WEB---")
    queries = state["search_terms"]
    all_results = []

    with DDGS() as ddgs:
      for search_term in queries:
          print(f"🔍 Searching DuckDuckGo for: {search_term}")
          results = ddgs.text(search_term, max_results=2)
          urls = [r["href"] for r in results if "href" in r]
          all_results.extend(urls)
          time.sleep(5)

    return {**state, "search_urls": all_results}

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state):
    print("---GRADING DOCUMENT RELEVANCE---")
    st.write("---GRADING DOCUMENT RELEVANCE---")
    task = state["task"]
    documents = state["indirect_documents"]

    llm = state["llm"]

    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance. \n
      Your job is to assess whether the retrieved content is for helping update, complete, or improve the original task. \n
      If the content contains information directly useful to STRICTLY updating the task, grade it as relevant.
      Give a binary score 'yes' or 'no' score to indicate whether the content is relevant to the task."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Existing Task: {task} \n\n\n Retrieved content: \n\n {content}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    # Score each doc
    filtered_docs = {}
    for url, doc in documents.items():
        score = retrieval_grader.invoke(
            {"task": task, "content": doc['text']}
        )
        grade = score.binary_score
        print('--> URL:', url)
        if grade == "yes":
            print("      ---GRADE: DOCUMENT RELEVANT---")
            filtered_docs[url] = doc
        else:
            print("      ---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    all_relevant_docs = filtered_docs | state["relevant_documents"]
    print('All relevant docs:', all_relevant_docs)
    return {**state, "relevant_documents": all_relevant_docs}

# Search Terms
class SearchTerms(BaseModel):
    """List of search terms based on a task"""
    terms: List[str] = Field(description="A list of concise search terms related to the task")

def generate_search_terms(state):
    task = state["task"]
    llm = state["llm"]
    structured_llm_grader = llm.with_structured_output(SearchTerms)

    # Prompt
    search_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates search terms based on a task description."),
        ("human", "Generate a list of 3 to 5 search terms that are most likely to help find most relevant and up to date information related to this task. \n\n\nTask: {task}")
    ])

    search_terms = search_prompt | structured_llm_grader
    result = search_terms.invoke({"task": task})
    print(result.terms)
    print('\n'*5)
    return {**state, "task": task, "search_terms": result.terms}

def get_canonical_url(url):
    """
    Visits a URL, follows any redirects, and returns the final destination URL.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=15)
        response.raise_for_status()
        final_url = response.url
        if final_url != url:
            None
            # print(f"URL redirected: {url} -> {final_url}")
        return final_url
    except requests.exceptions.RequestException as e:
        print(f"Could not resolve canonical URL for {url}: {e}")
        return url
    
def get_latest_wayback_snapshot(url):
    """
    Fetches the URL and date for the most recent snapshot of a given CANONICAL URL.
    Returns a dictionary containing the URL and a datetime object, or None if not found.
    """
    print(f"Searching archive for latest snapshot of: {url}")
    api_url = "http://web.archive.org/cdx/search/cdx"
    params = {'url': url, 'output': 'json', 'limit': -1, 'filter': 'statuscode:200', 'collapse': 'digest'}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data or len(data) < 2:
            print("No snapshots found for this URL.")
            return None
        
        # The first item in the list is headers, the rest are snapshots.
        # We take the first snapshot entry which is data[1].
        headers = data[0]
        snapshot_info_list = data[1]
        snapshot_info = dict(zip(headers, snapshot_info_list))
        
        timestamp = snapshot_info.get('timestamp')
        original_url = snapshot_info.get('original')
        
        if timestamp and original_url:
            # Parse the timestamp string (e.g., '20230815123045') into a datetime object
            snapshot_datetime = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            snapshot_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
            
            # Return a dictionary with both the URL and the parsed date
            return {
                "url": snapshot_url,
                "date": snapshot_datetime
            }
        else:
            print("Could not parse snapshot data from API response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during API request: {e}")
        return None

# def fetch_and_clean_html(url, source_name):
#     """
#     Fetches a URL and returns its cleaned, visible text content.
#     """
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     try:
#         print(f"Fetching content from {source_name} ({url})...")
#         response = requests.get(url, headers=headers, timeout=20)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'lxml')
#         for script_or_style in soup(["script", "style"]):
#             script_or_style.decompose()
#         return ' '.join(t.strip() for t in soup.stripped_strings)
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching {source_name} URL: {e}")
#         return None

def fetch_and_clean_html(url, source_name):
    """
    Fetches a URL, validates that it's HTML, and returns its cleaned, 
    visible text content.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        print(f"Fetching content from {source_name} ({url})...")
        response = requests.get(url, headers=headers, timeout=20)
        
        # 1. Check for HTTP errors (4xx or 5xx)
        response.raise_for_status()

        # 2. Check if the content is actually HTML before parsing
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            print(f"Warning: Content from {source_name} is not HTML (type: {content_type}). Skipping.")
            return None

        # 3. Use the lenient 'html.parser' and wrap in a try block for safety
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.decompose()
            return ' '.join(t.strip() for t in soup.stripped_strings)
        except Exception as e:
            # This will catch any unexpected errors during parsing
            print(f"Error parsing HTML from {source_name} ({url}): {e}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {source_name} URL ({url}): {e}")
        return None

def summarize_changes_with_llm(llm, snapshot_text, live_text):
    """
    Uses an LLM to provide a semantic summary of the differences between two texts.
    """
    print("\n--- Asking LLM to Summarize Changes ---")

    # Safety check: Truncate long texts to avoid exceeding token limits
    max_chars = 20000
    if len(snapshot_text) > max_chars:
        snapshot_text = snapshot_text[:max_chars] + " ... [content truncated]"
    if len(live_text) > max_chars:
        live_text = live_text[:max_chars] + " ... [content truncated]"

    # Create a clear prompt for the LLM
    prompt_template = """
    You are an expert web page analyst. Your job is to identify and summarize the key changes
    between two versions of a webpage's text content.

    Analyze the following two versions and provide a concise, bulleted summary of the substantive changes.
    Focus on changes to headings, paragraphs, contact information, eligibility rules, updated forms, or application steps.
    Ignore minor changes like whitespace, or tiny punctuation differences.
    If there are no meaningful changes, simply state that.

    --- OLD VERSION (Snapshot) ---
    {snapshot_text}

    --- NEW VERSION (Live Website) ---
    {live_text}

    --- SUMMARY OF CHANGES ---
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create a LangChain chain
    chain = prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({
            "snapshot_text": snapshot_text,
            "live_text": live_text
        })
        return summary
    except Exception as e:
        return f"An error occurred while communicating with the LLM: {e}"

def compare_live_with_snapshot(initial_url, llm):
    """
    Compares the current live version of a URL with its latest snapshot.
    Returns a dictionary containing the comparison dates, a summary of differences,
    and a boolean indicating if changes were found.
    """
    print(f"\n--- Starting comparison for: {initial_url} ---")

    today_date = datetime.now()
    # Format dates as strings for the final dictionary
    today_date_str = today_date.strftime('%Y-%m-%d')

    canonical_url = get_canonical_url(initial_url)
    snapshot_data = get_latest_wayback_snapshot(canonical_url)

    if not snapshot_data:
        print("Could not find a snapshot to compare against. Exiting.")
        return {
            "today_date": today_date_str,
            "snapshot_date": None,
            "summary": "No snapshot available for comparison.",
            "has_changed": False
        }

    snapshot_url = snapshot_data["url"]
    snapshot_date = snapshot_data["date"]
    snapshot_date_str = snapshot_date.strftime('%Y-%m-%d') # Format for dictionary

    print(f"Comparing dates:")
    print(f"  - Live Website Date: {today_date.strftime('%d %B %Y')}")
    print(f"  - Last Snapshot Date: {snapshot_date.strftime('%d %B %Y')}")
    print(f"Latest snapshot URL found: {snapshot_url}")

    raw_snapshot_url = snapshot_url.replace('/web/', '/web/', 1).replace('/http', 'id_/http', 1)
    live_text = fetch_and_clean_html(canonical_url, "Live Website")
    snapshot_text = fetch_and_clean_html(raw_snapshot_url, "Snapshot")

    if live_text is None or snapshot_text is None:
        print("Could not fetch content from one or both sources. Cannot compare.")
        return {
            "today_date": today_date_str,
            "today_url": canonical_url,
            "snapshot_date": snapshot_date_str,
            "snapshot_url": snapshot_url,
            "summary": "Could not fetch content from one or both sources.",
            "has_changed": False
        }

    if live_text == snapshot_text:
        print("\n✅ RESULT: No difference found between the live site and the latest snapshot.")
        return {
            "today_date": today_date_str,
            "today_url": canonical_url,
            "snapshot_date": snapshot_date_str,
            "snapshot_url": snapshot_url,
            "summary": "No difference found.", # A clear summary for no changes
            "has_changed": False
        }
    else:
        print("\n⚠️ RESULT: A difference was detected!")
        summary = summarize_changes_with_llm(llm, snapshot_text, live_text)
        print("\n--- LLM Summary of Differences ---")
        print(summary)
        return {
            "today_date": today_date_str,
            "today_url": canonical_url,
            "snapshot_date": snapshot_date_str,
            "snapshot_url": snapshot_url,
            "summary": summary,
            "has_changed": True
        }

def save_snapshot(url):
    """
    Requests a snapshot of a URL to be saved in the Wayback Machine.
    Returns the URL of the new snapshot if successful.
    """
    save_url = 'https://web.archive.org/save/' + url
    headers = {
        # A User-Agent is good practice to avoid being blocked
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # We don't want to follow the redirect, we want to capture its location
        response = requests.get(save_url, headers=headers, allow_redirects=False, timeout=1200)

        # A successful request will result in a 302 redirect
        if response.status_code == 302:
            # The URL of the new snapshot is in the 'Location' header
            archive_url = response.headers.get('Location')

            if archive_url:
                print(f"Snapshot successfully requested. Archive is available at:")
                print(archive_url)
                return archive_url
            else:
                print("Snapshot request was accepted (302), but Location header was not found.")
                return None
        # Sometimes a 200 is returned with a status page, not a direct snapshot
        elif response.status_code == 200:
            print("Received a 200 OK. This might be a status page. Check the content.")
            print("The most reliable indicator of a new snapshot is a 302 redirect.")
            return None
        else:
            print(f"Snapshot request failed with status code: {response.status_code}")
            print(f"Response Text: {response.text[:200]}") # Print first 200 chars of response
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return None

def wayback_machine_snapshots(state):
    print("---WAYBACK MACHINE ANALYSIS---")
    st.write("---WAYBACK MACHINE ANALYSIS---")
    wayback_machine_results = {}
    llm = state["llm"]
    task_urls = state.get("task_urls", [])
    
    # Check if this step is enabled in settings
    if 'run_settings' in state and state['run_settings'].get('wayback_machine_snapshots'):
        for url_to_check in task_urls:
            st.write(f'Analysis on {url_to_check}')
            # The function now returns a single dictionary with all info
            comparison_result = compare_live_with_snapshot(url_to_check, llm)
            st.json(comparison_result)
            
            # Store the entire result dictionary for the URL
            wayback_machine_results[url_to_check] = comparison_result
            
            # Check the 'has_changed' flag inside the result dictionary
            if comparison_result['has_changed'] and state['run_settings'].get('wayback_machine_snapshots_save'):
                save_snapshot(url_to_check)

    return {**state, "wayback_machine_results": wayback_machine_results}

class GraphState(TypedDict):
    llm: ChatGoogleGenerativeAI
    task: str # Task in markdown format
    run_settings: dict # Settings for the run, which features to run
    update_prompt: str # The user-editable prompt for proposing updates
    search_terms: List[str]
    task_urls: List[str]  # URLs embedded in the task
    search_urls: List[str]  # URLs retrieved from DuckDuckGo
    urls: List[str]  # Combined list used for scraping
    indirect_documents: dict
    relevant_documents: dict
    proposed_update: dict
    wayback_machine_results: dict

def build_graph(GraphState):

    workflow = StateGraph(GraphState)

    workflow.add_node("extract_task_urls", extract_task_urls)
    workflow.add_node("generate_search_terms", generate_search_terms)
    workflow.add_node("search_duckduckgo", search_duckduckgo)
    workflow.add_node("scrape_urls", scrape_urls)
    workflow.add_node("grade_indirect_content", grade_documents)
    workflow.add_node("propose_update", propose_update)
    workflow.add_node("wayback_machine_snapshots", wayback_machine_snapshots)


    workflow.set_entry_point("extract_task_urls")
    workflow.add_conditional_edges(
        "extract_task_urls",  # The source node
        should_search,        # The function that decides the next path
        {
            # The 'path_map': maps the return value of 'should_search' to the next node
            "generate_search_terms": "generate_search_terms",
            "scrape_urls": "scrape_urls"
        }
    )
    workflow.add_edge("generate_search_terms", "search_duckduckgo")
    workflow.add_edge("search_duckduckgo", "scrape_urls")
    workflow.add_edge("scrape_urls", "grade_indirect_content")
    workflow.add_edge("grade_indirect_content", "propose_update")
    workflow.add_edge("propose_update", "wayback_machine_snapshots")
    workflow.add_edge("wayback_machine_snapshots", END)

    app = workflow.compile()

    return app

def extract_additional_sources(main_data):
    try:
        url_pattern = re.compile(
                r'https?://[^\s<>"\'\]\),]+'
            )
        additional_sources = url_pattern.findall(main_data['data']['internal_note'])
        return additional_sources
    except:
        print('------WARNING: Internal note not found------')
        print('Code will still run')
        return []

