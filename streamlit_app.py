import streamlit as st
import json

import os

from pprint import pprint

# LangChain and Pydantic imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Imports for capturing print statements
import sys
from io import StringIO

from langgraph_funcs import *

# ==============================================================================
# START: STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🤖 AI Task Analysis and Update Agent")
st.markdown("This application analyzes a task, scrapes relevant web pages, and suggests updates to the task instructions based on the latest information.")

# --- API Key Input ---
with st.sidebar:
    st.header("Configuration")
    st.write("Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
    google_api_key = st.text_input("Enter your Google API Key", type="password")
    if not google_api_key:
        st.warning("Please enter your Google API Key to proceed.")
    os.environ['GOOGLE_API_KEY'] = google_api_key

    st.header("⚙️ Analysis Settings")
    direct_sources_enabled = st.toggle("Analyze URLs from Task Content", value=True, help="Extract and scrape URLs found directly within the task's name, description, and instructions.")
    indirect_sources_enabled = st.toggle("Search Web for External Sources", value=True, help="Generate search terms based on the task and find additional, potentially more up-to-date sources online.")
    wayback_enabled = st.toggle("Compare with Wayback Machine", value=True, help="Check each direct URL against its archive.org snapshot to detect changes.")
    wayback_save_enabled = st.toggle("Save New Snapshot if Changes Found", value=False, help="If a difference is detected, save a new snapshot to archive.org.", disabled=not wayback_enabled)

# --- File Uploaders ---
col1, col2 = st.columns(2)
with col1:
    main_data_file = st.file_uploader("1. Upload Main JSON File (with internal_note)", type="json")

with col2:
    content_data_file = st.file_uploader("2. Upload Content JSON File (with task details)", type="json")

DEFAULT_UPDATE_PROMPT = """
You are tasked with analyzing the website content to determine whether the provided instructions need to be updated.

Your goal is to:
1. Identify any discrepancies or outdated information in the instructions based on the website content.
2. Assess the severity of each discrepancy — classify it as either minor or major.
3. Suggest specific, actionable updates to the instructions, focusing only on the content that requires revision.
4. Clearly explain the reasoning behind each suggested change.
5. If applicable, propose general improvements to enhance the clarity, completeness, or level of detail in the instructions.

Only evaluate and suggest updates related to the content of the instructions — do not rewrite or reformat unrelated parts found in the website content.

**Important: Your response must be valid JSON matching this schema:**

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

Original Task:
{task}

Context from current sources:
{context}

Your response in valid json:
"""

with st.expander("📝 Customize Update Proposal Prompt"):
    custom_prompt = st.text_area(
        "Edit the prompt used by the AI agent to generate task updates. Ensure you keep the `{task}` and `{context}` placeholders.",
        value=DEFAULT_UPDATE_PROMPT,
        height=400,
        key="update_prompt_editor"
    )

# --- Run Button ---
if st.button("🚀 Run Analysis", disabled=(not main_data_file or not content_data_file or not google_api_key)):

    # --- Load and Process Data ---
    try:
        main_data = json.loads(main_data_file.getvalue().decode("utf-8"))
        content_data = json.loads(content_data_file.getvalue().decode("utf-8"))
        
        # Initialize LLM here after API key is set
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.9)
        # llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9)
    except Exception as e:
        st.error(f"Error loading JSON files or initializing the model: {e}")
        st.stop()

    
    additional_sources = extract_additional_sources(main_data)
    # Assemble run_settings from UI controls
    run_settings = {
        "direct_sources": direct_sources_enabled,
        "indirect_sources": indirect_sources_enabled,
        "wayback_machine_snapshots": wayback_enabled,
        "wayback_machine_snapshots_save": wayback_save_enabled,
        "additional_sources": additional_sources
    }
    
    # --- Capture Logs ---
    log_stream = StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_stream

    # status_tracker = StatusTracker() # Initialize our UI tracker
    # result = {}
    
    try:
        with st.spinner("Analyzing... This may take a few minutes."):
            
            # --- Build and Run the Graph ---
            print("--- INITIALIZING ANALYSIS ---")
            print("\nRun Settings:")
            pprint(run_settings)
            
            full_task_md = obtain_full_task_md(content_data)
            
            app = build_graph(GraphState)

            task_input = {
                "llm": llm, # Pass llm instance in the state
                "task": full_task_md,
                "run_settings": run_settings,
                "update_prompt": custom_prompt, # Pass the user-edited prompt
            }

            # for event in app.stream(task_input):
            #     current_node = list(event.keys())[0]
            #     current_state = event[current_node]
                
            #     # Update the UI based on the latest state
            #     status_tracker.update_from_graph_state(current_node, current_state)
                
            #     # Store the final result
            #     if current_node == "__end__":
            #         result = current_state
            with st.expander("Live Analysis"):
                result = app.invoke(task_input)
            print("\n✅ ANALYSIS COMPLETE")

    finally:
        # Restore stdout and get log content
        sys.stdout = original_stdout
        log_content = log_stream.getvalue()

    # --- Display Results ---
    st.success("Analysis Complete!")

    # Display the processed markdown task for reference
    with st.expander("View Full Task Content (as processed)"):
        # col1, col2 = st.columns(2)

        # with col1: 
        st.markdown(full_task_md)

    # Create tabs for the different result sections
    tab_updates, tab_wayback, tab_logs = st.tabs(["🤖 Proposed Updates", "🗄️ Wayback Analysis", "⚙️ Process Logs"])

    with tab_updates:
        proposed_update = result.get("proposed_update", {})
        
        if not proposed_update:
            st.warning("No updates were proposed by the agent.")
        else:
            st.header("Proposed Updates")
            
            for url, update_data in proposed_update.items():
                with st.expander(f"**Suggestions for: `{url}`**", expanded=True):

                    update_content = update_data.get("update", {})
                    discrepancies = update_content.get("discrepancies", [])
                    improvements = update_content.get("general_improvements", [])

                    if not discrepancies and not improvements:
                        st.info("No specific discrepancies or improvements were identified for this source.")
                        continue
                    
                    if discrepancies:
                        st.subheader("Discrepancies Found")
                        for i, item in enumerate(discrepancies):
                            st.markdown(f"**Reasoning:** {item.get('description_reasoning')}")
                            st.markdown(f"**Severity:** `{item.get('severity', 'N/A').upper()}`")
                            update = item.get('suggested_update', {})
                            if update.get('original_text') or update.get('revised_text'):
                                st.markdown("**Suggested Change:**")
                                col1, col2 = st.columns(2)
                                col1.text_area("Original Text", value=update.get('original_text', 'N/A'), height=150, disabled=True, key=f"dis_orig_{url}_{i}")
                                col2.text_area("Revised Text", value=update.get('revised_text', 'N/A'), height=150, disabled=True, key=f"dis_rev_{url}_{i}")
                            st.divider()

                    if improvements:
                        st.subheader("General Improvements")
                        for i, item in enumerate(improvements):
                            st.markdown(f"**Suggestion:** {item.get('description_reasoning')}")
                            update = item.get('suggested_update', {})
                            if update.get('original_text') or update.get('revised_text'):
                                col1, col2 = st.columns(2)
                                col1.text_area("Original Section", value=update.get('original_text', 'N/A'), height=150, disabled=True, key=f"imp_orig_{url}_{i}")
                                col2.text_area("Suggested Improvement", value=update.get('revised_text', 'N/A'), height=150, disabled=True, key=f"imp_rev_{url}_{i}")
                            st.divider()
            
            # --- Display Raw JSON ---
            st.subheader("📄 Raw JSON Output")
            json_string = json.dumps(proposed_update, indent=4)
            st.download_button(
                    label="Download Raw JSON",
                    file_name="proposed_update.json",
                    mime="application/json",
                    data=json_string,
                )
            with st.expander("Click to view the full JSON response"):
                st.json(proposed_update)

    with tab_wayback:
        # --- Display Wayback Machine Results ---
        st.header("Wayback Machine Change Analysis")
        wayback_results = result.get("wayback_machine_results", {})

        if not wayback_results:
            st.info("Wayback Machine analysis was not enabled or did not produce any results.")
        else:
            for url, data in wayback_results.items():
                with st.expander(f"**Analysis for: `{url}`**", expanded=True):

                    # Use columns for a clean date layout
                    col1, col2 = st.columns(2)
                    col1.metric(
                        label="Live Version Date",
                        value=data.get("today_date", "N/A")
                    )
                    col2.metric(
                        label="Last Snapshot Date",
                        value=data.get("snapshot_date", "Not Found")
                    )

                    today_url = data.get("today_url")
                    snapshot_url = data.get("snapshot_url")

                    if today_url:
                        st.markdown(f"🔗 **Live Website:** [{today_url}]({today_url})")
                    
                    if snapshot_url:
                        st.markdown(f"🗄️ **Archived Snapshot:** [{snapshot_url}]({snapshot_url})")

                    st.markdown("---") # Visual separator

                    # Display status and summary based on whether changes were detected
                    if data.get("has_changed"):
                        st.warning("⚠️ **Changes Detected!**")
                        st.subheader("Summary of Changes")
                        st.markdown(data.get("summary", "No summary provided."))
                    else:
                        # Display the reason if it wasn't a successful comparison
                        if "No difference found" not in data.get("summary", ""):
                            st.info(data.get("summary"))
                        else: 
                            st.success("✅ **No significant changes were detected.**")

            # --- Display Raw JSON ---
            st.subheader("📄 Raw JSON Output")
            json_string = json.dumps(wayback_results, indent=4)
            st.download_button(
                    label="Download Raw JSON",
                    file_name="wayback_results.json",
                    mime="application/json",
                    data=json_string,
                )
            with st.expander("Click to view the full JSON response"):
                st.json(wayback_results)

    with tab_logs:
        # --- Display Logs ---
        st.header("⚙️ Process Logs")
        st.text_area("Logs", log_content, height=400, help="Detailed logs from the analysis run.")

else:
    st.info("Please upload both JSON files and enter your API key to enable the 'Run Analysis' button.")
