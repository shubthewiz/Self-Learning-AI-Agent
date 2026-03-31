# 🤖 The Resilient Job-Search Agent 

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq%20Llama%203.3-orange)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Most job scrapers are "dumb"—they hit a login wall or a CAPTCHA and simply crash. 

This is an **Autonomous Job-Search Agent** designed with a **Reasoning Engine**. If it hits a roadblock on LinkedIn, it doesn't quit; it updates its internal memory, analyzes the failure, and dynamically pivots to a different platform (like Indeed) to finish the task.

---

## 📺 Demo
![Job Agent Demo](link-to-your-gif-or-video-here)
*The agent attempting LinkedIn, hitting a login block, and pivoting to Indeed.*

---

## ✨ Key Features
* **Logic Pivot:** When a platform blocks the agent, it uses LLM-based reasoning to choose the "safest next move."
* **Persistent Memory:** Uses a JSON-based state system to "remember" which platforms are currently inaccessible.
* **Deep Enrichment:** Instead of just grabbing titles, the agent opens individual job listings to extract full descriptions and requirements.
* **Intelligent Scoring:** Ranks job listings based on keyword relevance to your specific profile.
* **Terminal Dashboard:** Beautiful, real-time logging using the `Rich` library.

---

## 🛠️ Tech Stack
* **Brain:** `Llama-3.3-70b` (via Groq Cloud) for lightning-fast planning.
* **Automation:** `Playwright` for resilient browser interaction.
* **UI:** `Rich` for structured terminal output.
* **State Management:** JSON-based persistence.

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* A Groq API Key (Free tier works great!)

### 2. Installation
```bash
# Clone the repository
git clone 

# Install dependencies
pip install -r requirements.md

# Install Playwright browsers
playwright install chromium
