# NovaBank Predictive Model
### Analytics Methods and Frameworks — Quantic MSBA · May 2026
**Authors:** Mduduzi Ndlovu & Mukabalengu S. Mukuni

---

## Project summary
ML-powered campaign targeting framework that predicts which customers 
are likely to subscribe to a term deposit. Built using logistic regression, 
Random Forest and k-means customer segmentation on 45,211 customer records.

**Key result:** Targeting the top 20% of customers by model score captures 
78.2% of all subscribers — a 3.6× lift over random outreach — while 
reducing call volume by 52%.

---

## Files
| File | Description |
|------|-------------|
| `NovaBank Analytics Notebook.ipynb` | Full reproducible notebook (6 steps) |
| `Novabank_dashboard.py` | Streamlit interactive dashboard |
| `bank-full.csv` | Dataset (45,211 customer records) |
| `requirements.txt` | Python dependencies |

---

## How to run the notebook
1. Upload the `.ipynb` file and `bank-full.csv` to [Google Colab](https://colab.research.google.com)
2. Run all cells top to bottom

## How to run the dashboard
```bash
pip install -r requirements.txt
streamlit run Novabank_dashboard.py
```

---

## AI usage
Built with assistance from Claude (Anthropic) and ChatGPT— see the AI Usage Log 
in the notebook for details on what AI contributed.
