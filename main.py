from graph.graph import graph


def main():
    print("Hello from agentic-rag-langgraph!")
    response = graph.invoke(
        {"question": "What is Machine Learning way of training  LLM ?"}
    )

    print(f"Answer: {response["generation"]}")


if __name__ == "__main__":
    main()
