import pandas as pd
import requests
import time
import re
from tqdm import tqdm

def classify_companies_into_sectors(df, company_column='Company', batch_size=40, model="llama3"):
    """
    Extracts unique company names from a DataFrame and classifies them into sectors using Ollama.
    Returns a DataFrame with columns: 'Company', 'Sector'
    """
    unique_companies = df[company_column].dropna().unique().tolist()
    print(f"🔍 Found {len(unique_companies)} unique companies to classify.\n")

    all_results = {}

    for i in tqdm(range(0, len(unique_companies), batch_size), desc="Processing Batches"):
        batch = unique_companies[i:i+batch_size]
        batch_results = get_sectors_from_ollama_csv(batch, model)
        all_results.update(batch_results)
        time.sleep(1)  # Avoid overwhelming the API

    return pd.DataFrame(list(all_results.items()), columns=["Company", "Sector"])


def get_sectors_from_ollama_csv(companies, model="minstral"):
    """
    Classify companies into sectors using Ollama. Returns a dict {Company: Sector}.
    """
    prompt = (
    "You are an expert classification assistant who can effectively group companies into their correct business sectors.\n"
    "Some company names may appear in incomplete or abbreviated forms — for example:\n"
    "- 'Hind. Unilever' means 'Hindustan Unilever'\n"
    "- 'St Bk of India' means 'State Bank of India'\n"
    "- 'Reliance Industr' means 'Reliance Industries'\n"
    "- and many more similar abbreviations maybe seen.\n"
    "Please use your understanding to interpret and normalize such names correctly.\n\n"
    "Then Classify each company below into its most relevant business sector, if you are confused, put in unknown, no explanation. **from the fixed list below**.\n"
    "Only use these sector names — do not make up new ones.\n\n"
    "SECTOR LIST:\n"
    "- IT & Technology\n"
    "- Energy & Power\n"
    "- Cement & Construction\n"
    "- Finance & Banking\n"
    "- Pharmaceuticals & Healthcare\n"
    "- Automotive & Transport\n"
    "- Infrastructure & Construction\n"
    "- FMCG & Consumer Goods\n"
    "- Retail & Lifestyle\n"
    "- Telecom & Media\n"
    "- Travel & Hospitality\n"
    "- Metals & Mining\n"
    "- Agro & Chemicals\n"
    "- Education & Training\n"
    "- Real Estate\n"
    "- Others\n\n"
    "Return the result strictly in CSV format with exactly two columns: Company,Sector\n"
    "No explanation, no extra text. Every company must be listed exactly once.\n\n"
    + "\n".join(companies)
)



    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            content = response.json().get("response", "")
            return parse_csv_response(content, companies)
        else:
            print(f"❌ Error from Ollama API: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception calling Ollama API: {e}")

    return {company: "Unknown" for company in companies}


def parse_csv_response(csv_text, expected_companies):
    """
    Parses CSV-formatted text into a dictionary. Verifies all companies are covered.
    """
    lines = csv_text.strip().splitlines()
    result = {}

    for line in lines:
        if "," in line:
            parts = [p.strip().strip('"') for p in line.split(",", 1)]
            if len(parts) == 2:
                company, sector = parts
                result[company] = sector

    # Fallbacks for missing companies
    for company in expected_companies:
        if company not in result:
            result[company] = "Unknown"

    return result


def load_input_data():
    """
    Load company list from companies.csv or companies.txt fallback.
    """
    try:
        df = pd.read_csv(r"C:\Users\anshr\Downloads\mid.csv")
        print("📥 Loaded from companies.csv")
        return df
    except:
        try:
            with open("companies.txt", "r") as f:
                companies = [line.strip() for line in f if line.strip()]
            print("📥 Loaded from companies.txt")
            return pd.DataFrame({"Company": companies})
        except:
            print("⚠️ No input file found. Using example data.")
            return pd.DataFrame({
                "Company": [
                    "Apple", "JPMorgan Chase", "Exxon Mobil", "Pfizer", "Boeing",
                    "Microsoft", "Walmart", "Coca-Cola", "Amazon", "Merck"
                ]
            })


def main():
    df = load_input_data()
    result_df = classify_companies_into_sectors(df, batch_size=10)

    print("\n✅ Classification Complete. Top 10 Results:\n")
    print(result_df.head(10))

    result_df.to_csv("company_sectors_llama3.csv", index=False)
    print("\n💾 Saved to company_sectors.csv")

    print("\n📊 Top Sectors:")
    print(result_df["Sector"].value_counts().head(10))


if __name__ == "__main__":
    main()
