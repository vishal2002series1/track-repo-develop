# visualize_wf_001.py
from src.workflows.wf_001 import build_WF_001_graph

def generate_graph_image():
    print("🎨 Generating WF_001 Compiled Graph...")
    app = build_WF_001_graph()
    
    try:
        image_data = app.get_graph().draw_mermaid_png()
        file_name = "wf_0001_architecture.png"
        with open(file_name, "wb") as f:
            f.write(image_data)
            
        print(f"✅ Success! Open '{file_name}' to see your compiled workflow.")
    except Exception as e:
        print("⚠️ Could not generate PNG. Ensure you have network access for Mermaid.")
        print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    generate_graph_image()