from generator import generator
from retriever import retriever
from app import HistOracleApp
import os

# Instantiate and run the app
if __name__ == "__main__":
    # Set base root directory; this should be the dir that holds the main config, src, data directories
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Initiate app
    app = HistOracleApp(project_root=project_root)
    app.run()
