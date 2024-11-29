




# Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jschild01/Capstone.git
   ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up AWS Bedrock credentials in the `config/config.ini` file:
    ```bash
    [BedRock_LLM_API]
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_SESSION_TOKEN=your_session_token
    ```
4. Set up AWS Bedrock credentions in the `config/.env` file:
    ```bash
    CONFIG_FILE=config.ini
    ```
5. Preconfigured vector stores should be available with the following names and stores in the `../data/vector_stores` directory. Vector stores not available in the repository will not function in the application unless they are constructed and housed locally or in a hosted environment.
    - `vectorstore_all_250_instruct`
    - `vectorstore_all_1000_instruct`
    - `vectorstore_all_250_titan`
    - `vectorstore_sample_250_instruct` (available in the repo)
    - `vectorstore_sample_1000_instruct` (available in the repo)

# Usage
Run the following command in the terminal to execute the Streamlit application.
```bash
streamlit run src/main.py
```    
