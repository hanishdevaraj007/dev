from flask import Flask,request,render_template #lightweight
import pandas as pd #dataframework for reading files
import google.generativeai as genai 
import os
from dotenv import load_dotenv

#step 1: call api key model
load_dotenv()

app=Flask(__name__) #application starts herey

#configure model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-2.5-flash")

df=pd.read_csv("qa_data (1).csv")

#convert cvs into context text
context_text=""

for _,row in df.iterrows():
    context_text += f"Q: {row['question']}\nA: {row['answer']}\n\n"

def ask_gemini(query):
    prompt = f""" You are a Q&A assistant. Answer only using the following context text.
    If the answer it is not present then say: No relavant Q&A foiund.
    Context: {context_text}
    Question: {query}"""

    response = model.generate_content(prompt)
    
    return response.text.strip()

#connection between frontend and backend using ROUTE
@app.route("/",methods=["POST","GET"])

def home():
    answer = ""
    if request.method=="POST":
        query = request.form["query"]
        answer = ask_gemini(query)
    return render_template("index.html",answer=answer)

#run app
if __name__ == "__main__":
    app.run()
