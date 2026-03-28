import PIL
import gradio as gr
from src.Chatbot.graph import build_graph, run_pipeline
import argparse


def build_interface(dataset="chroma_db"):
    def run(message, history):
        image_path = message.get("files", None)
        if image_path:
            with open(image_path[0], "rb") as image_file:
                image = image_file.read()
        else:
            image = None
        query = message.get("text", "")
        pipeline = build_graph()
        response = run_pipeline(pipeline, query, image, dataset=dataset)
        return response["answer"]

    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the H-RAG chatbot interface.")
    parser.add_argument(
        "-d",
        "--chroma_db_path",
        type=str,
        default="chroma_db",
        help="Path to the ChromaDB database",
    )
    args = parser.parse_args()
    run = build_interface(dataset=args.chroma_db_path)
    gr.ChatInterface(
        fn=run,
        multimodal=True,
        textbox=gr.MultimodalTextbox(file_types=["image"], sources=["upload"]),
    ).launch()
