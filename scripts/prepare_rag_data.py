# scripts/prepare_rag_data.py

import os
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

def create_product_catalog(path="data/product_catalog.xlsx"):
    """
    Creates a dummy Excel file with a product catalog.
    """
    if os.path.exists(path):
        print(f"'{path}' already exists. Skipping creation.")
        return

    data = {
        'Product_ID': [f'ELE-{i:03d}' for i in range(1, 21)],
        'Name': [
            'Laptop Pro X', 'Smartphone G5', 'Wireless Headphones', '4K Ultra HD TV', 
            'Gaming Mouse', 'Mechanical Keyboard', 'Smartwatch Series 5', 'Bluetooth Speaker',
            'Digital Camera Z1', 'External SSD 1TB', 'Drone Explorer', 'VR Headset',
            'E-Reader Scribe', 'Portable Projector', 'Soundbar 2.1', 'Webcam HD 1080p',
            'USB-C Hub', 'Wireless Charger', 'Noise-Cancelling Earbuds', 'Fitness Tracker'
        ],
        'Price': [
            1299.99, 799.99, 199.99, 1499.99, 79.99, 129.99, 399.99, 99.99,
            649.99, 119.99, 899.99, 499.99, 249.99, 299.99, 179.99, 59.99,
            49.99, 39.99, 149.99, 89.99
        ],
        'Features': [
            '16GB RAM, 512GB SSD, Intel i7', '6.5" OLED, Triple Camera, 5G', '30-hour battery, ANC', 
            '65-inch, HDR10+, Smart TV', '16000 DPI, RGB lighting', 'Cherry MX Brown switches',
            'GPS, Heart Rate Monitor', 'Waterproof, 12-hour playback', '24MP, 4K Video',
            'USB 3.2, 550MB/s read speed', '4K camera, 30-min flight time', '120Hz display, integrated audio',
            '10.2" display, 300 ppi', '150-inch projection, 1080p', '120W, with Subwoofer',
            'Full HD, Autofocus, Stereo Mic', '7-in-1, HDMI, SD Card Reader', '15W Fast Charging, Qi-Certified',
            'Active Noise Cancellation, Transparency Mode', 'Step Counter, Sleep Tracking'
        ],
        'Return_Policy': ['30-day money-back guarantee'] * 20
    }

    df = pd.DataFrame(data)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_excel(path, index=False)
    print(f"Successfully created '{path}'")

def create_employee_handbook(path="data/employee_handbook.pdf"):
    """
    Creates a dummy PDF file with an employee handbook.
    """
    if os.path.exists(path):
        print(f"'{path}' already exists. Skipping creation.")
        return

    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Page 1: Title and Remote Work Policy
    story.append(Paragraph("Employee Handbook", styles['h1']))
    story.append(Paragraph("Remote Work Policy", styles['h2']))
    story.append(Paragraph("""
        Our company embraces a flexible work environment. This policy outlines the guidelines for remote work.
        All employees are eligible for remote work, subject to manager approval. Requests must be submitted
        at least two weeks in advance. Employees are expected to maintain their regular work hours and be
        available for communication during core business hours (10 AM - 4 PM).
    """, styles['Normal']))

    # Page 2: Remote Work Policy (Continued)
    story.append(Paragraph("Equipment and Security", styles['h3']))
    story.append(Paragraph("""
        The company will provide necessary equipment, including a laptop and monitor. Employees are
        responsible for maintaining a secure and ergonomic home office setup. All company data must be
        handled in accordance with our data security policies. Use of public Wi-Fi for sensitive
        work is strictly prohibited.
    """, styles['Normal']))

    # Page 3: Expense Reimbursement
    story.append(Paragraph("Expense Reimbursement", styles['h2']))
    story.append(Paragraph("""
        All work-related expenses must be pre-approved by your manager. To request reimbursement,
        submit an expense report with original receipts within 30 days of the purchase. Reimbursable
        expenses include travel, office supplies, and pre-approved software. Meals are generally not
        reimbursable unless for client entertainment or during approved travel.
    """, styles['Normal']))

    # Build the PDF
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.build(story)
    print(f"Successfully created '{path}'")


if __name__ == "__main__":
    create_product_catalog()
    create_employee_handbook()
    print("Dummy data generation complete.")
