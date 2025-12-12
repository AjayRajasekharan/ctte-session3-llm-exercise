# 📘 CTTE Session 3 – Lab Exercises 
### Using CodeSandbox (No Installation Needed)

Welcome to the lab environment for **Session 3** of the CTTE x UWC Career Catalyst program.  
This sandbox lets you complete your practice exercuses fully online, without installing Python on your laptop.

You will be able to access three components:

1. **Demo** (Streamlit App Demo)  
2. **Teaching Exercise** (For teaching basic and advance LLM concepts)
3. **Practice Exercise** (For students to practice)

Everything runs in **CodeSandbox**, inside a preconfigured virtual environment (VM).

---

## 🚀 1. Getting Started

### **Step 1 — Open the CodeSandbox link shared by your instructor**
This link loads a fully configured environment with Python and Jupyter Notebook preinstalled:

🔗 **[https://codesandbox.io/p/devbox/ctte-session3-homework-df48ql](https://codesandbox.io/p/github/AjayRajasekharan/ctte-session3-llm-exercise/main)**

### **Step 2 — Sign in to CodeSandbox (required for saving your work)**
Click **“Sign in”** at the top-right.  
You can use:
- Your **Google account**
- Your **GitHub account**
- Or create a simple CodeSandbox account

You must be signed in so that:
- You can **Fork** the project  
- Your work will **auto-save**  
- You can **publish to GitHub** at the end

### **Step 3 — Fork the sandbox**
Click the **Fork** button at the top-right.  
This creates **your own editable workspace** (your personal copy of the environment).

### **Step 4 — Run Setup Tasks to complete**
1. Go to the left hand side pane, and select the 4th icon from top 'CODESANDBOX'.
2. Select on the 'Run Setup Tasks', it should open a pane in the right hand side and install dependencies.


<img width="318" height="762" alt="image" src="https://github.com/user-attachments/assets/a10cc542-c376-4056-8b05-d441d017e030" />


3. This should enable the preview and allow you to open external tab in the bottom right corner pop-up.

<img width="803" height="899" alt="image" src="https://github.com/user-attachments/assets/0f74278d-ccda-49b4-872e-370106a143aa" />

CodeSandbox will automatically:
- Install dependencies using `pip install -r requirements.txt`
- Start **Jupyter Notebook** on port **8888**

Watch the logs in the terminal.  
Once you see *“Jupyter Notebook is running on port 8888”*, move to the next step.

---

## 📓 2. Opening the Jupyter Notebook

### **Step 1 — Open port 8888**
CodeSandbox will show a notification or a port badge: Jupyter Notebook running on port 8888

Click:

> **Open in external**

❗ *Do NOT use the small preview panel — always open the external browser tab.*

### **Step 2 — Open the notebook file**
In the Jupyter Notebook interface:

1. Navigate to the `Practice/` folder  
2. Click **`practice_exercise.ipynb`**  
3. Work through all the exercise cells and Tasks

Your progress is saved automatically in CodeSandbox. 

Don't forget to close the jupyter kernals by selecting the 'Shut Down All' or clicking the 'X' near the kernel 'Shut down kernel', so that it you conserve your CPU usage.

<img width="1232" height="506" alt="image" src="https://github.com/user-attachments/assets/30fb6db7-8351-4be8-954d-3f2595ef4305" />

---

## 🤖 3. Running Streamlit Application

Inside the repository, you will find:

Demo/llm_advance_demo.py -> Opens a separate window and runs using Streamlit UI

To run the Streamlit chatbot:
1. Open the **terminal** in CodeSandbox  
2. Run: streamlit run Demo/llm_advance_demo.py

_streamlit run_ _file_name.py_

---


## 💾 4. Publishing your Repository

To publish your Repository, you need to first sign in using your GitHub account.
After signing in :

### **Step 1 — Open the Export to GitHub**
Click the **Export to GitHub** icon on the left sidebar (it's the last icon that looks like the GitHub image).

<img width="301" height="921" alt="image" src="https://github.com/user-attachments/assets/e36d42ad-d8a6-44be-acda-a65d1c2eea65" />


### **Step 2 — Click “Create Repository”**
This will open a prompt that asks you to name the repository, workspace and privacy setting.

<img width="937" height="466" alt="image" src="https://github.com/user-attachments/assets/018d90e4-b60a-4368-988a-59ccb235de6c" />


### **Step 3 — Click 'Create repository'**
This will create and push a new repository to your GitHub account that you have connected.

It will also create a repository linked to your codeSandbox.

---

## 🛠 5. Troubleshooting Guide

### **Jupyter Notebook doesn’t open**
Make sure you clicked **“Open in browser”** for port **8888** (not the small preview window).

### **Notebook appears as raw JSON**
This happens if you open `.ipynb` inside the CodeSandbox editor.  
Instead, always open the notebook from the **Jupyter browser interface**.

### **Chatbot gives `ModuleNotFoundError`**
You may have:
- Opened a wrong file, or
- Not forked the full environment  
Reopen the sandbox using the instructor’s link → click **Fork** again.

### **“Publish Branch” keeps spinning forever**
Try pushing manually:

```bash
git push
```

If it still fails, wait 10–15 seconds and retry — CodeSandbox occasionally delays GitHub syncing.

### **My notebook changes are not visible**

Refresh the Jupyter tab.
Jupyter auto-saves frequently, but sometimes the browser tab lags behind.

### **Port 8888 does not appear**

Wait for the setup tasks to finish.
Once you see a log message like: 'Jupyter Notebook is running on port 8888' the port should become available.

