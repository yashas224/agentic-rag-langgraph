from langgraph.graph import END, START, StateGraph

from graph.chains import answer_grader, hallucination_grader
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


def checkForAnswerRevelanceAndHallucination(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    hallucination_response: hallucination_grader.GradeHallucinations = (
        hallucination_grader.hallucination_grader.invoke(
            {"documents": state["documents"], "generation": state["generation"]}
        )
    )

    if hallucination_response.binary_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        answer_check: answer_grader.AnswerCheck = answer_grader.answer_grader.invoke(
            {"question": state["question"], "answer": state["generation"]}
        )
        if answer_check.isRelated:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "wrong answer generation"


builder.add_conditional_edges(
    GRADE_DOCUMENTS, condition, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)

builder.add_edge(WEBSEARCH, GENERATE)
builder.add_conditional_edges(
    GENERATE,
    checkForAnswerRevelanceAndHallucination,
    {"useful": END, "not useful": WEBSEARCH, "wrong answer generation": GENERATE},
)
builder.add_edge(GENERATE, END)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
