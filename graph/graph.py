from langgraph.graph import END, START, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_document, retrieve, web_search
from graph.state import GraphState

print("Building a LangGraph Graph ")
builder = StateGraph(GraphState)

builder.add_node(GRADE_DOCUMENTS, grade_document.gradeDocNode)
builder.add_node(WEBSEARCH, web_search.webSearchNode)
builder.add_node(GENERATE, generate.generate_node)
builder.add_node(RETRIEVE, retrieve.retrieveNode)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)


def condition(state: GraphState) -> str:
    if state.get("web_search"):
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


builder.add_conditional_edges(
    GRADE_DOCUMENTS, condition, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)

builder.add_edge(WEBSEARCH, GENERATE)
builder.add_edge(GENERATE, END)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
