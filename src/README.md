# Key Files
- `generator.py`: Handles text generation processes.
- `retriever.py`: Handles data retrieval tasks.
- `app.py`: Main application logic

<img width="581" alt="retriever_diagram" src="https://github.com/user-attachments/assets/caf21441-162e-4a24-bb3a-2f7410528647">


# Usage
The application:
- Sets up the project directory structure.
- Adds relevant paths to the Python environment.
- Loads the generator, retriever, and application classes.
- Instantiates and runs the HistOracleApp.

Modify the directory paths in the script if the default structure does not fit your project setup.

Run the following command in the terminal from the root directory to execute the Streamlit application.
```bash
streamlit run src/main.py
``` 


# Customization
Update the following variables in `main.py` to reflect your custom directory structure:
```
data_dir = os.path.join(project_root, 'data')
demo_dir = os.path.join(project_root, 'demo')
vstore_dir = os.path.join(demo_dir, 'vector_stores')
config_dir = os.path.join(project_root, 'config')
src_dir = os.path.join(project_root, 'src')
components_dir = os.path.join(src_dir, 'components')
```

To add new components to the application, place the files in the `src/components/` directory and ensure they are imported as needed.



# License
This project is licensed under the MIT License. See the LICENSE file in the root directory for details.
