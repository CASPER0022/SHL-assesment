# Deployment Guide for Render

Follow these steps to deploy your SHL Assessment Recommender and get your public API URL.

### 1. Prepare your Code
Ensure the following files are in your repository:
- `main.py` (Contains the FastAPI app)
- `requirements.txt` (List of dependencies)
- `shl_catalogue.json` (The data source)
- `catalog_index_tfidf.pkl` (The pre-computed index for fast startup)
- `.env` (Optional for local, but **DO NOT** commit this to GitHub)

### 2. Push to GitHub
1. Create a new repository on [GitHub](https://github.com).
2. Initialize git in your local folder (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
3. Link to your GitHub repo and push:
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

### 3. Deploy on Render
1. Go to [Render.com](https://render.com) and log in.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub account and select your repository.
4. Configure the service:
   - **Name**: `shl-assessment-recommender` (or any unique name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **Advanced** to add Environment Variables:
   - Key: `GEMINI_API_KEY`
   - Value: `your_actual_api_key_here`

### 4. Verify Deployment
1. Wait for Render to finish the build and deploy.
2. Once "Live", copy the URL (e.g., `https://shl-assessment-recommender.onrender.com`).
3. Test your health check: Open `https://your-app.onrender.com/health` in your browser. You should see `{"status": "ok"}`.
4. Use your public URL to submit the assignment!

---

### Why we changed `main.py`?
Render assigns a random port to your app via the `PORT` environment variable. The code was updated to read this variable automatically:
```python
port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```
