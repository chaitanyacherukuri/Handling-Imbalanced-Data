"""
LangGraph workflow definition for the class imbalance agent.
"""
from typing import Dict, Optional, TypedDict, Any
from langgraph.graph import StateGraph, END, START

from nodes import (
    load_data_node,
    analyze_distribution_node,
    detect_imbalance_node,
    recommend_technique_node,
    apply_resampling_node,
    visualize_results_node,
    save_results_node,
    recommend_ml_algorithm_node
)

# Simplified state schema
class ImbalanceAgentState(TypedDict):
    # Core inputs
    file_path: str
    target_column: str
    output_dir: Optional[str]

    # Essential data and results (using Any for simplicity)
    data: Optional[Any]
    distribution: Optional[Dict]
    imbalance_info: Optional[Dict]
    recommended_technique: Optional[str]
    resampled_data: Optional[Any]
    applied_technique: Optional[str]
    original_shape: Optional[Any]
    resampled_shape: Optional[Any]
    visualization_paths: Optional[Dict]
    saved_path: Optional[str]
    ml_algorithm_recommendations: Optional[Dict]

    # Status tracking
    status: str
    message: Optional[str]
    error: Optional[str]

def create_workflow() -> StateGraph:
    """Create the simplified LangGraph workflow."""
    workflow = StateGraph(ImbalanceAgentState)

    # Add nodes
    nodes = [
        ("load_data", load_data_node),
        ("analyze_distribution", analyze_distribution_node),
        ("detect_imbalance", detect_imbalance_node),
        ("recommend_technique", recommend_technique_node),
        ("apply_resampling", apply_resampling_node),
        ("visualize_results", visualize_results_node),
        ("save_results", save_results_node),
        ("recommend_ml_algorithm", recommend_ml_algorithm_node)
    ]

    for name, node_func in nodes:
        workflow.add_node(name, node_func)

    # Simple linear workflow with error handling in decorators
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "analyze_distribution")
    workflow.add_edge("analyze_distribution", "detect_imbalance")
    workflow.add_edge("detect_imbalance", "recommend_technique")
    workflow.add_edge("recommend_technique", "apply_resampling")
    workflow.add_edge("apply_resampling", "visualize_results")
    workflow.add_edge("visualize_results", "save_results")
    workflow.add_edge("save_results", "recommend_ml_algorithm")
    workflow.add_edge("recommend_ml_algorithm", END)

    return workflow.compile()

def run_workflow(file_path: str, target_column: str, output_dir: Optional[str] = None) -> Dict:
    """Run the class imbalance workflow."""
    workflow = create_workflow()

    initial_state = {
        "file_path": file_path,
        "target_column": target_column,
        "output_dir": output_dir,
        "status": "initialized"
    }

    # Execute workflow and get final state
    final_state = None
    for state in workflow.stream(initial_state):
        final_state = state

    return final_state or initial_state
