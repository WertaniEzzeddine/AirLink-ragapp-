# AirLink-ragapp

## Setup Instructions

Follow these steps to set up and run the project:

### 1. **Create a Neo4j Database**
   - Go to the [Neo4j website](https://neo4j.com/) and create a new database.
   - Take note of your database URL, username, and password.

### 2. **Set Environment Variables**
   - In the `BackEnd` folder, you will find a `.env.sample` file.
   - Rename this file to `.env`.
   - Open the `.env` file and set the following environment variables with your Neo4j credentials:
     ```
    
     NEO4J_USERNAME=your_neo4j_username
     NEO4J_PASSWORD=your_neo4j_password
     ```
   - Replace `your_neo4j_database_url`, `your_neo4j_username`, and `your_neo4j_password` with your actual Neo4j database details.

### 3. **Get Your Groq API Key**
   - Visit the [Groq API console](https://console.groq.com/docs/api-keys).
   - Generate a new API key and keep it safe.

### 4. **Install Python Dependencies**
   - Navigate to the `BackEnd` directory.
   - Run the following command to install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

### 5. **Run the Backend**
   - After installing the requirements, run the FastAPI server using Uvicorn:
     ```bash
     uvicorn main:app --reload
     ```
   - The backend should now be running on `http://localhost:8000`.

### 6. **Setup Frontend**
   - Navigate to the `FrontEnd` directory.
   - Install the necessary dependencies by running:
     ```bash
     npm install
     ```
   - After installation, start the Angular development server:
     ```bash
     ng serve
     ```
   - The frontend should now be available at `http://localhost:4200`.

*## Additional Notes
- Make sure to update your `.env` file with correct API keys and credentials for the app to work properly.*
