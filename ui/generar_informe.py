# Ubicación: ProyectoIntermodular/ui/generar_informe.py
"""
Generador de informes PDF — Greenlight AI
Diseño oscuro profesional con semáforo visual, barra de score y layout premium.
"""

import io, re, math
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Paleta ────────────────────────────────────────────────────────────────────
PAGE_BG    = colors.HexColor("#060b14")
DARK_BG    = colors.HexColor("#0d1526")
MID_BG     = colors.HexColor("#111d30")
BORDER     = colors.HexColor("#1e2e45")
CYAN       = colors.HexColor("#00f2ff")
CYAN_DIM   = colors.HexColor("#006f80")
TEXT_MAIN  = colors.HexColor("#c8d6e5")
TEXT_MUTED = colors.HexColor("#6a7f95")
ROJO       = colors.HexColor("#ff4444")
AMBAR      = colors.HexColor("#ffaa00")
VERDE      = colors.HexColor("#22ff44")
ROJO_OFF   = colors.HexColor("#1f0505")
AMBAR_OFF  = colors.HexColor("#1f1400")
VERDE_OFF  = colors.HexColor("#011a05")
WHITE      = colors.white

def _csem(e): return {"rojo": ROJO, "ambar": AMBAR, "verde": VERDE}.get(e, CYAN)
def _hsem(e): return {"rojo": "ff4444", "ambar": "ffaa00", "verde": "22ff44"}.get(e, "00f2ff")


# ── Fondo + header de página ──────────────────────────────────────────────────
def _bg(canv, doc):
    W, H = A4
    canv.saveState()

    # Fondo negro-azul
    canv.setFillColor(PAGE_BG)
    canv.rect(0, 0, W, H, fill=1, stroke=0)

    # Barra superior degradada (simulada con rectángulos)
    for i, alpha in enumerate([1, 0.7, 0.4, 0.15]):
        canv.setFillColor(colors.Color(0, 0.95, 1, alpha=alpha * 0.9))
        canv.rect(0, H - (i+1)*1.8*mm, W, 1.8*mm, fill=1, stroke=0)

    # Texto en barra superior
    canv.setFillColor(PAGE_BG)
    canv.setFont("Helvetica-Bold", 6.5)
    canv.drawString(10*mm, H - 5.5*mm, "GREENLIGHT  ·  INFORME CLÍNICO DE RIESGO DE LESIÓN")
    canv.setFont("Helvetica", 6.5)
    canv.drawRightString(W - 10*mm, H - 5.5*mm, datetime.now().strftime("%d/%m/%Y   %H:%M"))

    # Línea inferior
    canv.setStrokeColor(CYAN_DIM)
    canv.setLineWidth(0.4)
    canv.line(18*mm, 14*mm, W - 18*mm, 14*mm)

    # Página
    canv.setFillColor(TEXT_MUTED)
    canv.setFont("Helvetica", 6)
    canv.drawCentredString(W / 2, 9*mm, f"Página {doc.page}")

    canv.restoreState()


# ── Semáforo dibujado ─────────────────────────────────────────────────────────
class Semaforo(Flowable):
    def __init__(self, estado, w=52, h=120):
        super().__init__()
        self.estado = estado
        self.width  = w
        self.height = h

    def draw(self):
        c   = self.canv
        cw  = self.width
        bw  = 32
        bh  = 100
        bx  = (cw - bw) / 2
        by  = (self.height - bh) / 2

        # Poste
        c.setFillColor(colors.HexColor("#151515"))
        c.roundRect(cw/2 - 2.5, by - 10, 5, 12, 2, fill=1, stroke=0)

        # Sombra carcasa
        c.setFillColor(colors.Color(0, 0, 0, alpha=0.5))
        c.roundRect(bx+2, by-2, bw, bh, 8, fill=1, stroke=0)

        # Carcasa
        c.setFillColor(colors.HexColor("#141414"))
        c.setStrokeColor(colors.HexColor("#2a2a2a"))
        c.setLineWidth(0.8)
        c.roundRect(bx, by, bw, bh, 8, fill=1, stroke=1)

        # Tornillos
        c.setFillColor(colors.HexColor("#2a2a2a"))
        for tx, ty in [(bx+5, by+6), (bx+bw-5, by+6),
                       (bx+5, by+bh-6), (bx+bw-6, by+bh-6)]:
            c.circle(tx, ty, 1.8, fill=1, stroke=0)

        r  = 9.5
        cx = bx + bw / 2
        luces = [
            (cx, by + bh - 16, ROJO,  ROJO_OFF,  self.estado == "rojo"),
            (cx, by + bh - 50, AMBAR, AMBAR_OFF, self.estado == "ambar"),
            (cx, by + bh - 84, VERDE, VERDE_OFF, self.estado == "verde"),
        ]
        for lx, ly, col_on, col_off, activa in luces:
            fill = col_on if activa else col_off
            if activa:
                # Glow exterior
                for rad, alpha in [(r+9,0.06),(r+6,0.12),(r+3,0.22)]:
                    c.setFillColor(colors.Color(fill.red, fill.green, fill.blue, alpha=alpha))
                    c.circle(lx, ly, rad, fill=1, stroke=0)
            # Luz
            c.setFillColor(fill)
            c.setStrokeColor(colors.Color(0,0,0,alpha=0.6))
            c.setLineWidth(0.5)
            c.circle(lx, ly, r, fill=1, stroke=1)
            # Reflejo
            if activa:
                c.setFillColor(colors.Color(1,1,1,alpha=0.3))
                c.ellipse(lx-3, ly+3, lx+2, ly+r-1, fill=1, stroke=0)


# ── Barra de progreso de score ────────────────────────────────────────────────
class BarraScore(Flowable):
    def __init__(self, score, color, w, h=14):
        super().__init__()
        self.score  = score   # 0.0-1.0
        self.color  = color
        self.width  = w
        self.height = h

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        r = h / 2

        # Fondo barra
        c.setFillColor(BORDER)
        c.roundRect(0, 0, w, h, r, fill=1, stroke=0)

        # Relleno
        fill_w = max(h, w * self.score)
        c.setFillColor(self.color)
        c.roundRect(0, 0, fill_w, h, r, fill=1, stroke=0)

        # Brillo
        c.setFillColor(colors.Color(1,1,1,alpha=0.12))
        c.roundRect(2, h*0.55, fill_w - 4, h*0.35, r*0.4, fill=1, stroke=0)

        # Texto del porcentaje
        pct = f"{round(self.score*100,1)}%"
        c.setFillColor(PAGE_BG if self.score > 0.15 else TEXT_MAIN)
        c.setFont("Helvetica-Bold", 7.5)
        tx = fill_w * 0.5
        c.drawCentredString(tx, h*0.25, pct)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _p(txt, **kw):
    base = dict(fontName="Helvetica", fontSize=9, textColor=TEXT_MAIN,
                leading=13, spaceAfter=2)
    base.update(kw)
    name = f"s{abs(hash(str(sorted(kw.items()))))%99999}"
    return Paragraph(txt, ParagraphStyle(name, **base))

def _s(n=3): return Spacer(1, n*mm)

def _hr(color=CYAN, thick=0.5, sb=3, sa=5):
    return HRFlowable(width="100%", thickness=thick, color=color,
                      spaceBefore=sb, spaceAfter=sa)

def _tbase():
    return [
        ("GRID",          (0,0),(-1,-1), 0.3, BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ]

def _card(content_rows, col_widths, extra_style=None):
    t = Table(content_rows, colWidths=col_widths)
    style = _tbase() + [("BACKGROUND",(0,0),(-1,-1),DARK_BG)]
    if extra_style:
        style += extra_style
    t.setStyle(TableStyle(style))
    return t


# ── Función principal ─────────────────────────────────────────────────────────
def generar_informe_pdf(
    *, nombre_jugador, posicion, fecha=None,
    acwr, distancia_m, dist_hd, sprints, acelerac, tiempo_juego,
    fc_max, horas_sueno, fatiga_subj,
    decision, estado_color, risk_score,
    razonamiento, tiempo_recuperacion, fuente,
    diagnostico_pdf="", reporte_fisio="",
) -> bytes:

    buf       = io.BytesIO()
    W, H      = A4
    mg        = 17*mm
    uw        = W - 2*mg
    fecha_str = fecha or datetime.now().strftime("%d/%m/%Y %H:%M")
    csem      = _csem(estado_color)
    hsem      = _hsem(estado_color)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=mg, rightMargin=mg,
        topMargin=13*mm, bottomMargin=20*mm,
        title=f"Greenlight — {nombre_jugador}",
        author="Greenlight AI",
    )
    story = []

    # ── CABECERA ──────────────────────────────────────────────────────────────
    logo_row = Table(
        [[_p("🟢  GREENLIGHT", fontName="Helvetica-Bold", fontSize=26,
              textColor=CYAN, alignment=TA_CENTER, leading=30, spaceAfter=1)],
         [_p("INFORME CLÍNICO DE RIESGO DE LESIÓN", fontName="Helvetica",
              fontSize=7.5, textColor=TEXT_MUTED, alignment=TA_CENTER,
              leading=10, spaceAfter=0)]],
        colWidths=[uw]
    )
    logo_row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), DARK_BG),
        ("TOPPADDING",    (0,0),(-1,0),  14),
        ("BOTTOMPADDING", (0,1),(-1,-1), 12),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("LINEBELOW",     (0,-1),(-1,-1),2, CYAN),
    ]))
    story += [logo_row, _s(5)]

    # ── FICHA JUGADOR ─────────────────────────────────────────────────────────
    story.append(_p("● DATOS DEL JUGADOR", fontName="Helvetica-Bold", fontSize=7.5,
                    textColor=CYAN, leading=10, spaceBefore=0, spaceAfter=4))

    def lbl(t): return _p(t, fontName="Helvetica-Bold", fontSize=7.5,
                           textColor=TEXT_MUTED, leading=11)
    def val(t): return _p(t, fontName="Helvetica", fontSize=9,
                           textColor=TEXT_MAIN, leading=12)

    ficha = Table([
        [lbl("NOMBRE / ID"), val(nombre_jugador), lbl("POSICIÓN"),  val(posicion)],
        [lbl("FECHA"),       val(fecha_str),       lbl("FUENTE"),   val(fuente)],
    ], colWidths=[uw*p for p in [0.17,0.33,0.17,0.33]])
    ficha.setStyle(TableStyle(_tbase() + [
        ("BACKGROUND",(0,0),(0,-1), MID_BG),
        ("BACKGROUND",(2,0),(2,-1), MID_BG),
        ("BACKGROUND",(1,0),(1,-1), DARK_BG),
        ("BACKGROUND",(3,0),(3,-1), DARK_BG),
    ]))
    story += [ficha, _s(5)]

    # ── DIAGNÓSTICO PRINCIPAL ─────────────────────────────────────────────────
    story.append(_p("● DIAGNÓSTICO PRINCIPAL", fontName="Helvetica-Bold", fontSize=7.5,
                    textColor=CYAN, leading=10, spaceBefore=0, spaceAfter=4))

    acwr_hex = "ff4444" if acwr>1.5 else "ffaa00" if acwr>1.3 else "22ff44"

    col_s = uw * 0.19
    col_k = uw * 0.405

    def _kpi_block(label1, val1_html, label2, val2_html, col_w):
        return Table([
            [_p(label1, fontName="Helvetica-Bold", fontSize=7, textColor=TEXT_MUTED,
                alignment=TA_CENTER, leading=9)],
            [_p(val1_html, fontName="Helvetica-Bold", fontSize=20,
                alignment=TA_CENTER, leading=24)],
            [_s(1)],
            [_hr(BORDER, 0.3, sb=0, sa=0)],
            [_s(1)],
            [_p(label2, fontName="Helvetica-Bold", fontSize=7, textColor=TEXT_MUTED,
                alignment=TA_CENTER, leading=9)],
            [_p(val2_html, fontName="Helvetica-Bold", fontSize=20,
                alignment=TA_CENTER, leading=24)],
        ], colWidths=[col_w])

    ki = _kpi_block(
        "ESTADO",         f'<font color="#{hsem}">{decision}</font>',
        "SCORE RIESGO",   f'<font color="#00f2ff">{round(risk_score*100,1)}%</font>',
        col_k
    )
    ki.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK_BG),
                             ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))

    kd = _kpi_block(
        "TIEMPO DE BAJA",  f'<font color="#{hsem}">{tiempo_recuperacion}</font>',
        "ACWR",            f'<font color="#{acwr_hex}">{acwr:.2f}</font>',
        col_k
    )
    kd.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK_BG),
                             ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))

    diag = Table([[Semaforo(estado_color, col_s, 120), ki, kd]],
                 colWidths=[col_s, col_k, col_k])
    diag.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), DARK_BG),
        ("GRID",          (0,0),(-1,-1), 0.3, BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("LINEABOVE",     (0,0),(-1,0),  2.5, csem),
        ("LINEBELOW",     (0,-1),(-1,-1),2.5, csem),
    ]))
    story += [diag, _s(3)]

    # Barra visual del score
    barra = BarraScore(risk_score, csem, uw, h=13)
    story += [barra, _s(5)]

    # ── RAZONAMIENTO IA ───────────────────────────────────────────────────────
    story.append(_p("● RAZONAMIENTO IA", fontName="Helvetica-Bold", fontSize=7.5,
                    textColor=CYAN, leading=10, spaceBefore=0, spaceAfter=4))
    rt = Table([[_p(razonamiento or "Sin razonamiento disponible.",
                    fontName="Helvetica", fontSize=8.5,
                    textColor=TEXT_MAIN, leading=13)]],
               colWidths=[uw])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), DARK_BG),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("TOPPADDING",    (0,0),(-1,-1), 9),
        ("BOTTOMPADDING", (0,0),(-1,-1), 9),
        ("LINELEFT",      (0,0),(0,-1),  3.5, csem),
    ]))
    story += [rt, _s(5)]

    # ── MÉTRICAS ──────────────────────────────────────────────────────────────
    story.append(_p("● MÉTRICAS DE ENTRADA", fontName="Helvetica-Bold", fontSize=7.5,
                    textColor=CYAN, leading=10, spaceBefore=0, spaceAfter=4))

    def th(t): return _p(t, fontName="Helvetica-Bold", fontSize=7.5,
                          textColor=CYAN, leading=11)

    cw_m = [uw*p for p in [0.28,0.22,0.28,0.22]]
    met = Table([
        [th("VARIABLE"),             th("VALOR"),               th("VARIABLE"),           th("VALOR")],
        [lbl("Distancia Total"),     val(f"{distancia_m:,} m"), lbl("FC Máx Sesión"),     val(f"{fc_max}%")],
        [lbl("Dist. HD >21 km/h"),   val(f"{dist_hd:,} m"),    lbl("Horas de Sueño"),    val(f"{horas_sueno} h")],
        [lbl("Sprints >25 km/h"),    val(str(sprints)),         lbl("Fatiga Subjetiva"),  val(f"{fatiga_subj}/10")],
        [lbl("Aceleraciones"),       val(str(acelerac)),        lbl("Minutos / semana"),  val(str(tiempo_juego))],
    ], colWidths=cw_m)
    met.setStyle(TableStyle(_tbase() + [
        ("BACKGROUND",(0,0),(-1,0),  MID_BG),
        ("LINEBELOW", (0,0),(-1,0),  1, CYAN),
        ("BACKGROUND",(0,1),(0,-1),  MID_BG),
        ("BACKGROUND",(2,1),(2,-1),  MID_BG),
        ("BACKGROUND",(1,1),(1,-1),  DARK_BG),
        ("BACKGROUND",(3,1),(3,-1),  DARK_BG),
    ]))
    story += [met, _s(5)]

    # ── REPORTE FISIO ─────────────────────────────────────────────────────────
    if reporte_fisio.strip():
        story.append(_p("● REPORTE DEL FISIOTERAPEUTA", fontName="Helvetica-Bold",
                        fontSize=7.5, textColor=CYAN, leading=10, spaceBefore=0, spaceAfter=4))
        txt = reporte_fisio[:1500] + ("…" if len(reporte_fisio)>1500 else "")
        ft = Table([[_p(txt, fontName="Helvetica", fontSize=8.5,
                        textColor=TEXT_MUTED, leading=13)]],
                   colWidths=[uw])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), DARK_BG),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
            ("RIGHTPADDING",  (0,0),(-1,-1), 12),
            ("TOPPADDING",    (0,0),(-1,-1), 9),
            ("BOTTOMPADDING", (0,0),(-1,-1), 9),
            ("LINELEFT",      (0,0),(0,-1),  2.5, TEXT_MUTED),
        ]))
        story += [ft, _s(5)]

    # ── DIAGNÓSTICO NARRATIVO PDF ─────────────────────────────────────────────
    if diagnostico_pdf.strip():
        story.append(_hr(CYAN, 0.4))
        story.append(_p("● ANÁLISIS NARRATIVO DEL INFORME PDF", fontName="Helvetica-Bold",
                        fontSize=7.5, textColor=CYAN, leading=10, spaceBefore=4, spaceAfter=4))
        limpio = re.sub(r'\*\*(.+?)\*\*', r'\1', diagnostico_pdf)
        limpio = re.sub(r'\*(.+?)\*',     r'\1', limpio)
        limpio = limpio[:2500] + ("…" if len(limpio)>2500 else "")
        nt = Table([[_p(limpio, fontName="Helvetica", fontSize=8.5,
                        textColor=TEXT_MAIN, leading=13)]],
                   colWidths=[uw])
        nt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), DARK_BG),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
            ("RIGHTPADDING",  (0,0),(-1,-1), 12),
            ("TOPPADDING",    (0,0),(-1,-1), 9),
            ("BOTTOMPADDING", (0,0),(-1,-1), 9),
            ("LINELEFT",      (0,0),(0,-1),  3.5, CYAN),
        ]))
        story += [nt, _s(5)]

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(_hr(TEXT_MUTED, 0.3, sb=6, sa=4))
    story.append(_p(
        f"Generado por Greenlight AI  ·  {fecha_str}  ·  "
        "Este informe es una herramienta de apoyo a la decisión clínica "
        "y no sustituye el criterio médico profesional.",
        fontName="Helvetica", fontSize=6.5, textColor=TEXT_MUTED,
        alignment=TA_CENTER, leading=10
    ))

    doc.build(story, onFirstPage=_bg, onLaterPages=_bg)
    return buf.getvalue()