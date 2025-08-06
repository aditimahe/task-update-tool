import streamlit as st


from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from langgraph_funcs import *

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
