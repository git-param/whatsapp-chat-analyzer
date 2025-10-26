# 📊 WhatsApp Chat Analyzer

A powerful Streamlit web app to analyze WhatsApp chat exports. Get deep insights into your conversations with statistics, timelines, activity heatmaps, word clouds, emoji analysis, and sentiment detection.

---

## 🚀 Features

- **File Summary:** Quick stats on messages, words, media, and links.
- **User Selection:** Analyze overall or individual user activity.
- **Timelines:** Interactive monthly and daily message trends.
- **Activity Analysis:** Most active days/months and weekly heatmaps.
- **Wordcloud & Common Words:** Visualize most used words.
- **Emoji Analysis:** Top emojis and their usage.
- **Sentiment Analysis:** Detect positive, negative, and neutral messages, with keyword and dangerous message detection.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/WhatsappChat_Analyzer.git
   cd WhatsappChat_Analyzer
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Export your WhatsApp chat** (without media) from your phone.
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
3. **Open the web browser** (usually at `http://localhost:8501`).
4. **Upload your chat `.txt` file** and explore the analysis!

---

## 📁 Project Structure

```
WhatsappChat_Analyzer/
│
├── app.py                # Main Streamlit app
├── preprocessor.py       # Chat preprocessing functions
├── helper.py             # Analysis helper functions
├── sentiment.py          # Sentiment analysis logic
├── requirements.txt      # Python dependencies
├── logo.png              # App logo (optional)
└── README.md             # Project documentation
```

---

## 📝 Notes

- Only supports WhatsApp chat exports in `.txt` format.
- For best results, export chats **without media**.
- Sentiment analysis uses a pre-trained ML model (see `sentiment.py`).

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key packages:
- streamlit
- matplotlib
- seaborn
- plotly
- pandas
- numpy
- scikit-learn
- wordcloud
- emoji

---

## 🙏 Credits

Developed by Param Dholakia.  
Inspired by the need for better chat insights!

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
