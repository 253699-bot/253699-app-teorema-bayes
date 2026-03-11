"""PDF generation service for analysis reporting."""

from __future__ import annotations

from fpdf import FPDF


def _normalize_text(text: str) -> str:
    """Normalize text replacing unsupported characters for FPDF latin-1 base or just encode it."""
    # FPDF handles latin-1 encoding nicely. 
    # Decode keeping valid european accents
    try:
        return text.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return "".join([c if ord(c) < 256 else "?" for c in text])


def crear_reporte_pdf(metricas: dict, insights_texto: str) -> bytes:
    """Generates a PDF analysis report and returns it as bytes."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", size=18, style="B")
    pdf.set_text_color(21, 50, 67) # #153243 corporate color
    title = _normalize_text("Reporte de Análisis Bayesiano")
    pdf.cell(200, 10, txt=title, ln=1, align="C")
    pdf.ln(10)
    
    # Metrics Table
    pdf.set_font("Arial", size=14, style="B")
    pdf.set_text_color(40, 75, 99) # #284B63 Secondary corporate
    pdf.cell(200, 10, txt=_normalize_text("Resumen de Métricas de Desempeño"), ln=1, align="L")
    pdf.ln(2)
    
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)
    for key, value in metricas.items():
        if isinstance(value, float):
            line = f"- {key.capitalize()}: {value:.2%}"
        else:
            line = f"- {key.capitalize()}: {value}"
        
        pdf.cell(200, 8, txt=_normalize_text(line), ln=1, align="L")
        
    pdf.ln(10)
    
    # AI Insights
    pdf.set_font("Arial", size=14, style="B")
    pdf.set_text_color(40, 75, 99)
    pdf.cell(200, 10, txt=_normalize_text("Conclusiones de la Inteligencia Artificial"), ln=1, align="L")
    pdf.ln(2)
    
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)
    
    clean_text = _normalize_text(insights_texto)
    pdf.multi_cell(0, 8, clean_text)
    
    # Generate Output bytes
    pdf_string = pdf.output(dest='S')
    return pdf_string.encode('latin-1')
