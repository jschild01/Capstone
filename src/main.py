import os
import sys

# Set base root directory
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

# Import your app
from app import HistOracleApp

# Create and run the app
app = HistOracleApp(project_root=project_root)
app.run()

# Note: No if __name__ == "__main__" guard is needed for Streamlit apps
# as Streamlit handles the execution context