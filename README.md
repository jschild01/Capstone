




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

# Usage
Run the following command in the terminal to execute the Streamlit application.
```bash
streamlit run src/main.py
```    
