import pandas as pd
from fpdf import FPDF
from flask import send_file
import os

# Load the Excel file once
disease_df = pd.read_excel("model/leafs.xlsx")

def create_disease_pdf(leaf_name):
    # Filter the row matching the leaf/disease name
    row = disease_df[disease_df["Leaf Names"] == leaf_name]

    if row.empty:
        return None

    row = row.iloc[0]  # Get the first match

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Disease Report: {leaf_name}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Leaf Name: {row['Leaf Names']}")
    pdf.multi_cell(0, 8, f"Moisture Level: {row['Moisture Level']}")
    pdf.multi_cell(0, 8, f"Water Level: {row['Water Level']}")
    pdf.multi_cell(0, 8, f"Causes: {row['Causes']}")
    pdf.multi_cell(0, 8, f"Supplements: {row['Supplements to Add']}")
    
    # Save PDF temporarily
    pdf_file = os.path.join("static", "uploads", f"{leaf_name}_report.pdf")
    pdf.output(pdf_file)
    return pdf_file
