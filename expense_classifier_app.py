import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- Preprocessing Function ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'inr', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- Expanded Sample Training Data ----------------
expanded_data = {
    'message': [
        # Food
        "INR 250 spent at Domino's", "INR 1500 spent at BigBasket", "INR 300 spent at Zomato", 
        "INR 450 spent at Swiggy", "INR 200 spent at McDonald's", "INR 1200 spent at Grofers", 
        "INR 800 spent at Amazon Pantry", "INR 350 spent at KFC", "INR 600 spent at Barista", 
        "INR 1000 spent at Starbucks",

        # Bills
        "INR 1200 paid for electricity bill", "INR 500 paid for mobile recharge", 
        "INR 800 paid for DTH recharge", "INR 1500 paid for broadband bill", 
        "INR 2000 paid for gas cylinder", "INR 1000 paid for water bill", 
        "INR 1500 paid for landline bill", "INR 1200 paid for insurance premium", 
        "INR 800 paid for credit card bill", "INR 2000 paid for loan EMI",

        # Shopping
        "INR 5000 spent at Flipkart", "INR 3000 spent at Myntra", "INR 1500 spent at Ajio", 
        "INR 2000 spent at Snapdeal", "INR 1000 spent at Tata CLiQ", "INR 2500 spent at Croma", 
        "INR 1200 spent at Shoppers Stop", "INR 1800 spent at Reliance Digital", 
        "INR 2200 spent at Decathlon", "INR 1500 spent at Lifestyle",

        # Travel
        "INR 500 spent on Uber ride", "INR 1500 spent on Ola ride", 
        "INR 3000 spent on train ticket", "INR 5000 spent on flight booking", 
        "INR 2000 spent on hotel booking", "INR 800 spent on bus ticket", 
        "INR 1000 spent on metro card recharge", "INR 1500 spent on car rental", 
        "INR 2500 spent on travel insurance", "INR 1200 spent on taxi fare",

        # Rent
        "INR 15000 paid for monthly rent", "INR 2000 paid for society maintenance", 
        "INR 5000 paid for house cleaning", "INR 1000 paid for parking charges", 
        "INR 3000 paid for electricity bill (rented house)", "INR 2000 paid for water charges", 
        "INR 1500 paid for internet charges", "INR 1000 paid for security charges", 
        "INR 2500 paid for repair charges", "INR 1200 paid for garbage collection",

        # Income
        "INR 50000 credited as salary", "INR 2000 credited as freelance payment", 
        "INR 10000 credited as bonus", "INR 15000 credited as part-time job payment", 
        "INR 3000 credited as cashback", "INR 5000 credited as gift money", 
        "INR 2000 credited as rent income", "INR 1000 credited as investment return", 
        "INR 1500 credited as refund", "INR 2500 credited as affiliate income",

        # Entertainment
        "INR 500 spent on movie tickets", "INR 1000 spent on Netflix subscription", 
        "INR 800 spent on Amazon Prime subscription", "INR 600 spent on Spotify subscription", 
        "INR 1500 spent on gaming", "INR 1200 spent on concert tickets", 
        "INR 2000 spent on event tickets", "INR 1000 spent on amusement park", 
        "INR 800 spent on clubbing", "INR 1500 spent on sports event",

        # Others
        "INR 500 spent on education", "INR 1000 spent on healthcare", 
        "INR 2000 spent on gifts", "INR 800 spent on charity", 
        "INR 1500 spent on pet care", "INR 1200 spent on home improvement", 
        "INR 1000 spent on personal care", "INR 2000 spent on subscriptions", 
        "INR 1500 spent on hobbies", "INR 1200 spent on miscellaneous"
    ],
    'category': [
        'Food']*10 + ['Bills']*10 + ['Shopping']*10 + ['Travel']*10 +
        ['Rent']*10 + ['Income']*10 + ['Entertainment']*10 + ['Others']*10
}

df = pd.DataFrame(expanded_data)
df['message'] = df['message'].apply(preprocess)

# ---------------- Train ML Model ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['category']
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ---------------- Streamlit Web App ----------------
st.set_page_config(page_title="Expense Classifier", layout="centered")
st.title("💳 Personal Finance Expense Classifier")

# ----- User Input -----
username = st.text_input("👤 Enter your name to start:", value="")

if username:
    st.markdown("### 📝 Paste your transaction messages (one per line):")
    messages_input = st.text_area("📩 Messages", placeholder="e.g.\nINR 500 spent on groceries\nPaid electricity bill INR 1200")

    if st.button("🔍 Analyze Messages"):
        messages = [m.strip() for m in messages_input.strip().split("\n") if m.strip()]
        if messages:
            processed = [preprocess(msg) for msg in messages]
            predictions = model.predict(vectorizer.transform(processed))

            result_df = pd.DataFrame({
                'User': username,
                'Message': messages,
                'Category': predictions
            })

            # Save to user-specific CSV
            filename = f"{username.lower().replace(' ', '_')}_transactions.csv"
            if os.path.exists(filename):
                existing = pd.read_csv(filename)
                result_df = pd.concat([existing, result_df], ignore_index=True)
            result_df.to_csv(filename, index=False)

            # Show result table
            st.markdown("### 📄 Categorized Transactions")
            st.dataframe(result_df)

            # Show bar chart
            st.markdown("### 📊 Expense Breakdown by Category")
            fig, ax = plt.subplots()
            sns.countplot(data=result_df, x='Category', palette='Set2', order=result_df['Category'].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter at least one message.")
else:
    st.info("ℹ️ Please enter your name to begin.")
