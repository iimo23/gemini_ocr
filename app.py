from flask import Flask, render_template, request, jsonify, url_for
import google.generativeai as genai
import os
from dotenv import load_dotenv
from markupsafe import escape
import markdown
import mimetypes

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

ALLOWED_MIME_TYPES = {
    "application/pdf","text/plain", "text/html", "text/csv", "text/xml", "text/rtf",
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/heic", "image/heif",
}

def allowed_file(filename):
    # Check if the file has one of the allowed extensions
    mime_type = mimetypes.guess_type(filename)[0]
    return mime_type in ALLOWED_MIME_TYPES


system_instruction = "You are a highly accurate and adaptable data extraction expert specializing in arabic invoice images. Given an invoice image, meticulously extract and return all relevant data in JSON format. Ensure the output is comprehensive, error-free, and formatted correctly, regardless of the image's layout or format. Adapt to different styles and structures to provide the most accurate and complete data extraction. If the image quality is poor, apply image enhancement techniques to improve clarity."

prompt = """Analyze the following image of an invoice and extract key details. Read both arabic and english data and compare between them to extract the most accurate data. Please respond only in JSON format, without additional explanations or notes. The JSON object should include the following fields:

    invoice_number: The invoice number from the document.
    invoice_date: The date the invoice was issued.
    currency: The currency used in the invoice.
    payment_terms: The payment terms specified on the invoice.
    supplier_name: Name of the company or person who issued the invoice.
    supplier_address: Address of the supplier, if available.
    supplier_vat: Supplier VAT number or tax number.
    customer_name: Name of the person or organization being billed.
    customer_address: Address of the customer, if available.
    customer_vat: Customer VAT number or tax number.
    line_items: A list of items being billed, where each item includes:
        id: Item ID or Item No.
        item_description_arabic: Item name or description in Arabic.
        item_description_english: Item name or description in English.
        quantity: Quantity of the item.
        uom: Unit of Measure.
        unit_price: Price per unit.
        total_price: Total price for the item.
        tax_percentage_per_item: Percentage of tax applied to the item, if specified.
        total_price_with_tax: Total price for the item after tax.
    subtotal: Subtotal amount for all line items before tax.
    tax: Amount of tax applied to the subtotal, if specified.
    total_amount_due: The total amount due on the invoice.


Return only a complete JSON object with these fields filled out based on the data in the invoice image."""


@app.route("/", methods=["GET","POST"])
def index():
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=system_instruction,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
        ),
    )

    gemini_response = ""
    uploaded_file_url = ""
    try:
        if request.method == "POST":
            file = request.files.get("file")
            if file:
                if allowed_file(file.filename):
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                    file.save(file_path)
                    # tesseract_response = pytesseract.image_to_string(file_path, lang="eng+ara")
                    file_to_upload = genai.upload_file(file_path)
                    response = model.generate_content([file_to_upload, "\n\n", prompt])
                    gemini_response = response.text
                    gemini_response = markdown.markdown(gemini_response)
                else:
                    gemini_response = "File type not allowed. Please upload a file of type: " + ", ".join(ALLOWED_MIME_TYPES)
            else:
                response = model.generate_content(prompt)
                gemini_response = response.text
                gemini_response = markdown.markdown(gemini_response)
    except Exception as e:
        # Handle errors and set an appropriate error message
        gemini_response = f"An error occurred: {str(e)}"
    return render_template("index.html", gemini_response=gemini_response, uploaded_file_url=uploaded_file_url)
