
```markdown

```
# How to use this Repository

## Step 1: Checkout a New Branch
```bash
git checkout -b user/pmallick/hack-01
```

## Step 2: Set Up a Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate
```

## Step 3: Install Dependencies
1. **Check installed packages (should be empty):**
    ```bash
    pip freeze
    ```
2. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Verify installed packages:**
    ```bash
    pip freeze
    ```

## Step 4: Run the Application
Run the application in the terminal under the `myenv` development environment:
```bash
python app.py
```

You should see the following output:
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://<your machine IP>:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```

## Step 5: Access the Application
Open your web browser and navigate to:
- http://127.0.0.1:5000
- http://<your machine IP>:5000
```
