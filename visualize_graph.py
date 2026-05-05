# visualize_graph.py
from src.agents.graph import build_aeon_graph

def generate_graph_image():
    print("🎨 Generating AEON Agent Graph...")
    app = build_aeon_graph()
    
    try:
        # LangGraph's internal function to generate a Mermaid architecture diagram
        image_data = app.get_graph().draw_mermaid_png()
        
        file_name = "aeon_architecture.png"
        with open(file_name, "wb") as f:
            f.write(image_data)
            
        print(f"✅ Success! Open '{file_name}' to see your multi-agent workflow architecture.")
    except Exception as e:
        print("⚠️ Could not generate PNG. Printing ASCII graph instead:")
        print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    generate_graph_image()