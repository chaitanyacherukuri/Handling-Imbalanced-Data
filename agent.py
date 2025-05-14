"""
LangGraph workflow definition for the class imbalance agent.
"""
from typing import Dict, List, Any, Tuple, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
import pandas as pd

from nodes import (
    load_data_node,
    analyze_distribution_node,
    detect_imbalance_node,
    recommend_technique_node,
    apply_resampling_node,
    visualize_results_node,
    save_results_node
)

# Define the state schema
class ImbalanceAgentState(TypedDict):
    # Input parameters
    file_path: str
    target_column: str
    output_dir: Optional[str]

    # Data
    data: Optional[pd.DataFrame]
    columns: Optional[List[str]]
    num_samples: Optional[int]

    # Analysis results
    distribution: Optional[Dict]
    imbalance_info: Optional[Dict]

    # Recommendation
    recommendation: Optional[str]
    recommended_technique: Optional[str]

    # Resampling results
    resampled_data: Optional[pd.DataFrame]
    applied_technique: Optional[str]
    original_shape: Optional[Tuple]
    resampled_shape: Optional[Tuple]

    # Visualization
    visualization_paths: Optional[Dict]

    # Output
    saved_path: Optional[str]

    # Status
    status: str
    message: Optional[str]
    error: Optional[str]

def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for the class imbalance agent.

    Returns:
        StateGraph: The workflow graph
    """
    # Create a new graph
    workflow = StateGraph(ImbalanceAgentState)

    # Add nodes to the graph
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("analyze_distribution", analyze_distribution_node)
    workflow.add_node("detect_imbalance", detect_imbalance_node)
    workflow.add_node("recommend_technique", recommend_technique_node)
    workflow.add_node("apply_resampling", apply_resampling_node)
    workflow.add_node("visualize_results", visualize_results_node)
    workflow.add_node("save_results", save_results_node)

    # Define the edges (workflow)
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "analyze_distribution")
    workflow.add_edge("analyze_distribution", "detect_imbalance")
    workflow.add_edge("detect_imbalance", "recommend_technique")
    workflow.add_edge("recommend_technique", "apply_resampling")
    workflow.add_edge("apply_resampling", "visualize_results")
    workflow.add_edge("visualize_results", "save_results")
    workflow.add_edge("save_results", END)

    # Add conditional edges for error handling
    def check_status(state: ImbalanceAgentState) -> str:
        """Check the status of the current node execution."""
        if state.get("status") == "failed":
            return "error"
        return "continue"

    workflow.add_conditional_edges(
        "load_data",
        check_status,
        {
            "error": END,
            "continue": "analyze_distribution"
        }
    )

    workflow.add_conditional_edges(
        "analyze_distribution",
        check_status,
        {
            "error": END,
            "continue": "detect_imbalance"
        }
    )

    workflow.add_conditional_edges(
        "detect_imbalance",
        check_status,
        {
            "error": END,
            "continue": "recommend_technique"
        }
    )

    workflow.add_conditional_edges(
        "recommend_technique",
        check_status,
        {
            "error": END,
            "continue": "apply_resampling"
        }
    )

    workflow.add_conditional_edges(
        "apply_resampling",
        check_status,
        {
            "error": END,
            "continue": "visualize_results"
        }
    )

    workflow.add_conditional_edges(
        "visualize_results",
        check_status,
        {
            "error": END,
            "continue": "save_results"
        }
    )

    # Compile the graph
    return workflow.compile()

def run_workflow(file_path: str, target_column: str, output_dir: Optional[str] = None) -> Dict:
    """
    Run the class imbalance workflow.

    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        output_dir: Directory to save outputs (optional)

    Returns:
        Dictionary with the final state of the workflow
    """
    # Create the workflow
    workflow = create_workflow()

    # Initialize the state
    initial_state = {
        "file_path": file_path,
        "target_column": target_column,
        "output_dir": output_dir,
        "status": "initialized",
        "message": "Workflow initialized"
    }

    # Run the workflow
    for state in workflow.stream(initial_state):
        current_node = state.get("current_node", "")
        status = state.get("status", "")
        message = state.get("message", "")

        if current_node:
            print(f"Executing node: {current_node}")
            print(f"Status: {status}")
            print(f"Message: {message}")
            print("-" * 50)

    # Get the final state
    final_state = state

    return final_state
