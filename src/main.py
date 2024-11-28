import os
import sys

# Instantiate and run the app
if __name__ == "__main__":
    # Set base root directory; this should be the dir that holds the main config, src, data directories
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, 'data')
    demo_dir = os.path.join(project_root, 'demo')
    vstore_dir = os.path.join(demo_dir, 'vector_stores')
    config_dir = os.path.join(project_root, 'config')
    src_dir = os.path.join(project_root, 'src')
    components_dir = os.path.join(src_dir, 'components')

    # Add paths to required component/app files
    sys.path.append(demo_dir)
    sys.path.append(data_dir)
    sys.path.append(src_dir)
    sys.path.append(components_dir)

    # Get file classes/components
    from generator import generator
    from retriever import retriever
    from app import HistOracleApp

    # Initiate app
    app = HistOracleApp(project_root=project_root)
    app.run()
