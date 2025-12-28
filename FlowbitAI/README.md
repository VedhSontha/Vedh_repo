# Invoice Processing System (AI-Powered)

This system automates invoice processing by learning from user corrections. It uses a **Memory Intelligence** layer to remember vendor details and corrections, applying them to future invoices.

## Features
- **Invoice Ingestion**: Upload text-based invoice files.
- **Auto-Correction**: Applies learned patterns (e.g., typos like `Vndr_lnc` -> `Vendor Inc`) automatically.
- **Fuzzy Matching**: Intelligent matching for unseen typos.
- **Learning Loop**: User corrections in the Review interface are saved to memory.

## How It Works (The "AI" Logic)
This is not just a regex parser. It implements an **Adaptive Memory System**:
1.  **Extraction**: Simple pattern matching extracts raw data.
2.  **Memory Recall**: The system checks its `flowbit.db` for similar past errors using **Fuzzy Logic** (Sequence Matching > 80% similarity).
3.  **Self-Correction**: If it finds a match (e.g., "Vndr_Inc" is similar to a known error meant to be "Vendor Inc"), it **automatically swaps** the value before you even see it.

## Demo
We have included a script to generate sample data for a demo video:
1. Run `python demo_runner.py`.
2. This will generate `demo_invoice_1_teach.txt` and `demo_invoice_2_test.txt`.
3. Follow the on-screen script to record your video.

## Setup

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Initialize Database**:
   ```bash
   python -m src.init_db
   ```

## Usage

### 1. Run the App
```bash
streamlit run app.py
```

### 2. The Workflow
1.  **Upload**: Go to the main dashboard and upload a `.txt` invoice (e.g., `Invoice #1 from Vndr_Inc, Total: $500`).
2.  **Process**: Click "Process Invoice". If the vendor is unknown, it stays "Pending".
3.  **Review & Teach**:
    - Go to **Review Invoices** (sidebar).
    - Correct the data (e.g., change `Vndr_Inc` to `Vendor Inc`).
    - Click **Approve & Learn**.
4.  **Verify Intelligence**:
    - Upload a *new* invoice with the same bad vendor name.
    - The system will now auto-correct it and mark it as "Auto-Corrected".

## Testing
Run the automated verification suite:
```bash
python tests/test_core.py
python tests/test_optimization.py
```
