#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate mlquantify_full_documentation.pdf

A comprehensive reference covering every quantification method, solver,
representation and loss in mlquantify. Each entry follows the two-surface
model defined in mlquantify_doc_standard.pdf:

    * API Docstring  -> interface (no math)
    * User Guide     -> theory (all math lives here)

Content is grounded in the source papers stored in Google Drive
(papers/Quantification and papers/Quantification/Methods).

Math notation uses ASCII-safe symbols and reportlab <sub>/<super> markup,
never Unicode sub/superscripts (those render as black boxes in the built-in
Type-1 fonts).
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether, NextPageTemplate,
)
from reportlab.platypus.tableofcontents import TableOfContents

OUT = "mlquantify_full_documentation.pdf"

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
INK      = colors.HexColor("#1b2733")
SLATE    = colors.HexColor("#33475b")
ACCENT   = colors.HexColor("#0b6e99")   # mlquantify blue
ACCENT2  = colors.HexColor("#117a8b")
LIGHT    = colors.HexColor("#eef3f7")
RULE     = colors.HexColor("#c7d3dd")
CODEBG   = colors.HexColor("#f4f6f8")
MUTED    = colors.HexColor("#5b6b7a")

# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------
ss = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

STYLES = {
    "CoverTitle": S("CoverTitle", fontName="Helvetica-Bold", fontSize=30,
                    leading=34, textColor=INK, alignment=TA_CENTER, spaceAfter=10),
    "CoverSub": S("CoverSub", fontName="Helvetica", fontSize=13.5, leading=19,
                  textColor=SLATE, alignment=TA_CENTER, spaceAfter=6),
    "CoverMeta": S("CoverMeta", fontName="Helvetica", fontSize=10.5, leading=15,
                   textColor=MUTED, alignment=TA_CENTER),
    "Part": S("Part", fontName="Helvetica-Bold", fontSize=22, leading=26,
              textColor=colors.white, alignment=TA_LEFT),
    "PartNo": S("PartNo", fontName="Helvetica-Bold", fontSize=11, leading=13,
                textColor=colors.white, alignment=TA_LEFT, spaceAfter=4),
    "Family": S("Family", fontName="Helvetica-Bold", fontSize=15.5, leading=19,
                textColor=ACCENT, spaceBefore=8, spaceAfter=2),
    "FamilyIntro": S("FamilyIntro", fontName="Helvetica-Oblique", fontSize=9.6,
                     leading=14, textColor=SLATE, spaceAfter=6, alignment=TA_JUSTIFY),
    "Entry": S("Entry", fontName="Helvetica-Bold", fontSize=13.5, leading=16,
               textColor=INK, spaceBefore=6, spaceAfter=1),
    "EntryTag": S("EntryTag", fontName="Helvetica-Oblique", fontSize=8.6,
                  leading=11, textColor=MUTED, spaceAfter=4),
    "Surface": S("Surface", fontName="Helvetica-Bold", fontSize=9.2, leading=12,
                 textColor=colors.white, spaceBefore=4, spaceAfter=4),
    "Sub": S("Sub", fontName="Helvetica-Bold", fontSize=9.6, leading=12,
             textColor=ACCENT2, spaceBefore=5, spaceAfter=1),
    "Body": S("Body", fontName="Helvetica", fontSize=9.3, leading=13.2,
              textColor=INK, alignment=TA_JUSTIFY, spaceAfter=3),
    "Bullet": S("Bullet", fontName="Helvetica", fontSize=9.3, leading=13,
                textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=1.5),
    "Param": S("Param", fontName="Helvetica", fontSize=8.8, leading=12,
               textColor=INK, leftIndent=10, spaceAfter=1.5),
    "Opt": S("Opt", fontName="Helvetica", fontSize=8.2, leading=10.8,
             textColor=SLATE, leftIndent=20, bulletIndent=8, spaceBefore=0.5,
             spaceAfter=0.5),
    "Math": S("Math", fontName="Courier", fontSize=8.8, leading=12.4,
              textColor=colors.HexColor("#0a3d56"), leftIndent=12,
              backColor=CODEBG, borderColor=RULE, borderWidth=0,
              spaceBefore=2, spaceAfter=3),
    "Code": S("Code", fontName="Courier", fontSize=8.0, leading=10.6,
              textColor=INK, leftIndent=8, backColor=CODEBG, spaceBefore=2,
              spaceAfter=3),
    "Ref": S("Ref", fontName="Helvetica", fontSize=8.4, leading=11.4,
             textColor=SLATE, leftIndent=12, spaceAfter=1.5),
    "TOCFamily": S("TOCFamily", fontName="Helvetica-Bold", fontSize=10,
                   leading=14, textColor=INK, spaceBefore=4),
    "TOCItem": S("TOCItem", fontName="Helvetica", fontSize=9, leading=12.5,
                 textColor=SLATE, leftIndent=10),
    "Note": S("Note", fontName="Helvetica-Oblique", fontSize=9.0, leading=12.6,
              textColor=SLATE, alignment=TA_JUSTIFY, spaceAfter=3),
}

# ----------------------------------------------------------------------------
# Flowable helpers
# ----------------------------------------------------------------------------
def para(txt, st="Body"):
    return Paragraph(txt, STYLES[st])

def bullets(items, st="Bullet"):
    return [Paragraph(f"&bull;&nbsp; {it}", STYLES[st]) for it in items]

def math_block(lines):
    """Render monospace math lines inside a soft-shaded table cell."""
    inner = [Paragraph(ln.replace(" ", "&nbsp;"), STYLES["Math"]) for ln in lines]
    t = Table([[inner]], colWidths=[16.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t

def code_block(lines):
    inner = [Paragraph(ln.replace(" ", "&nbsp;"), STYLES["Code"]) for ln in lines]
    t = Table([[inner]], colWidths=[16.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfcfd")),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t

def surface_banner(label, color):
    t = Table([[Paragraph(label, STYLES["Surface"])]], colWidths=[16.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t

def rule(space_before=2, space_after=4):
    return HRFlowable(width="100%", thickness=0.5, color=RULE,
                      spaceBefore=space_before, spaceAfter=space_after)

# Bookmark / outline support -------------------------------------------------
_bm = {"n": 0}
class Anchor(Paragraph):
    """A zero-height paragraph that registers a PDF outline entry."""
    def __init__(self, key, title, level):
        super().__init__("", STYLES["Body"])
        self.key, self.title, self.level = key, title, level
    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=(self.level > 0))

def anchor(title, level):
    _bm["n"] += 1
    return Anchor(f"bm{_bm['n']}", title, level)

print("framework defined")

# ----------------------------------------------------------------------------
# Cover + front matter
# ----------------------------------------------------------------------------
def cover():
    fl = [Spacer(1, 3.3 * cm)]
    bar = Table([[""]], colWidths=[3.2 * cm], rowHeights=[0.16 * cm])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
    fl += [bar, Spacer(1, 0.5 * cm)]
    fl.append(para("mlquantify", "CoverTitle"))
    fl.append(para("Methods, Solvers, Representations &amp; Losses", "CoverTitle"))
    fl.append(Spacer(1, 0.4 * cm))
    fl.append(para("Comprehensive Documentation &mdash; API Docstrings &amp; User Guide",
                   "CoverSub"))
    fl.append(para("Every quantifier, optimisation backend, feature representation and "
                   "loss function, documented across the two-surface model.", "CoverSub"))
    fl.append(Spacer(1, 1.1 * cm))
    fl.append(HRFlowable(width="45%", thickness=0.7, color=RULE,
                         spaceBefore=2, spaceAfter=10, hAlign="CENTER"))
    fl.append(para("Version 0.3.1", "CoverMeta"))
    fl.append(para("Author: Luiz Fernando Luth Junior", "CoverMeta"))
    fl.append(para("Theory grounded in the source papers "
                   "(papers/Quantification &amp; papers/Quantification/Methods)", "CoverMeta"))
    fl.append(PageBreak())
    return fl

def how_to_read():
    fl = [anchor("How to read this document", 0),
          para("How to read this document", "Family"), rule()]
    fl.append(para(
        "This reference follows the <b>two-surface model</b> defined in the mlquantify "
        "documentation standard. Every entry is documented twice, for two different readers:",
        "Body"))
    data = [
        [Paragraph("<b>Surface</b>", STYLES["Param"]),
         Paragraph("<b>Audience</b>", STYLES["Param"]),
         Paragraph("<b>Content</b>", STYLES["Param"])],
        [Paragraph("API Docstring", STYLES["Param"]),
         Paragraph("Developer using the library", STYLES["Param"]),
         Paragraph("Interface: summary, parameters, attributes, examples, paper reference. "
                   "<b>No mathematics.</b>", STYLES["Param"])],
        [Paragraph("User Guide", STYLES["Param"]),
         Paragraph("Researcher learning the method", STYLES["Param"]),
         Paragraph("Theory: problem formulation, mathematical objective, algorithm, "
                   "assumptions, relationships.", STYLES["Param"])],
    ]
    t = Table(data, colWidths=[3.0 * cm, 4.6 * cm, 8.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    # recolor header cells text
    for c in range(3):
        data[0][c].style = ParagraphStyle("h", parent=STYLES["Param"],
                                           textColor=colors.white, fontName="Helvetica-Bold")
    fl += [Spacer(1, 3), t, Spacer(1, 8)]
    fl.append(para("<b>The golden rule.</b> Mathematical formulations, derivations and "
                   "pseudocode appear only in the User Guide blocks (shaded). The API "
                   "Docstring blocks describe the interface in words. Throughout, "
                   "<i>p</i> denotes a class-prevalence vector on the simplex "
                   "(p<sub>i</sub> &ge; 0, sum p<sub>i</sub> = 1), a hat (p&#770; or "
                   "<i>p</i>-hat) marks an estimate, &#39;tr&#39; the training set and "
                   "&#39;te&#39; the test set.", "Note"))
    fl.append(Spacer(1, 6))
    fl.append(para("Each method also names the <b>distributional shift</b> it targets. "
                   "Almost all assume <b>prior probability shift</b>: the class priors "
                   "p(y) change between training and test, while the class-conditional "
                   "densities p(x|y) stay fixed. This is the assumption under which the "
                   "corrections below are unbiased.", "Body"))
    fl.append(PageBreak())
    return fl

def part_divider(no, title, blurb):
    fl = [NextPageTemplate("body"), anchor(f"Part {no}: {title}", 0)]
    band = Table([[Paragraph(f"PART {no}", STYLES["PartNo"]),
                   ]], colWidths=[16.0 * cm])
    head = Table(
        [[Paragraph(f"PART {no}", STYLES["PartNo"])],
         [Paragraph(title, STYLES["Part"])]],
        colWidths=[16.0 * cm])
    head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 16),
        ("BOTTOMPADDING", (0, 1), (0, 1), 16),
    ]))
    fl = [Spacer(1, 1.2 * cm), head, Spacer(1, 0.5 * cm),
          para(blurb, "FamilyIntro")]
    return fl

# ----------------------------------------------------------------------------
# Entry renderer
# ----------------------------------------------------------------------------
def kv_table(rows):
    """Two-column name/description table for parameters & attributes.

    A description may be a plain string, or a ``(main, [(option, text), ...])``
    tuple. In the tuple form the main sentence is followed by an indented
    bullet list describing each allowed option.
    """
    data = []
    for name, desc in rows:
        if isinstance(desc, tuple):
            main, options = desc
            cell = [Paragraph(main, STYLES["Param"])]
            cell += [Paragraph(f"&bull;&nbsp; <font face='Courier'>{opt}</font> &mdash; {txt}",
                               STYLES["Opt"]) for opt, txt in options]
        else:
            cell = Paragraph(desc, STYLES["Param"])
        data.append([Paragraph(f"<font face='Courier'><b>{name}</b></font>", STYLES["Param"]),
                     cell])
    t = Table(data, colWidths=[3.7 * cm, 12.3 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e3e9ee")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t

def render_entry(e):
    fl = []
    head = [anchor(f"{e['name']} — {e['title']}", 2),
            para(f"{e['name']} &mdash; {e['title']}", "Entry"),
            para(f"{e['family']} family &nbsp;|&nbsp; "
                 f"<font face='Courier'>{e['imp']}</font>", "EntryTag")]
    # ---- API surface
    head.append(surface_banner("API DOCSTRING&nbsp;&nbsp;&middot;&nbsp;&nbsp;interface", SLATE))
    head.append(para("Summary", "Sub"))
    head.append(para(e["summary"], "Body"))
    fl.append(KeepTogether(head))

    fl.append(para("Description", "Sub"))
    fl.append(para(e["description"], "Body"))

    fl.append(para("Parameters", "Sub"))
    fl.append(kv_table(e["params"]))

    fl.append(para("Attributes", "Sub"))
    fl.append(kv_table(e["attrs"]))

    if e.get("notes"):
        fl.append(para("Notes", "Sub"))
        fl.append(para(e["notes"], "Note"))

    if e.get("see_also"):
        fl.append(para("See Also", "Sub"))
        fl.append(para(f"<font face='Courier'>{e['see_also']}</font>", "Body"))

    fl.append(para("Examples", "Sub"))
    fl.append(code_block(e["example"]))

    # ---- User Guide surface
    fl.append(Spacer(1, 2))
    fl.append(surface_banner("USER GUIDE&nbsp;&nbsp;&middot;&nbsp;&nbsp;theory", ACCENT2))
    fl.append(para("Problem formulation", "Sub"))
    fl.append(para(e["problem"], "Body"))

    fl.append(para("Algorithm / objective", "Sub"))
    alg = e["algorithm"]
    if alg.get("text"):
        fl.append(para(alg["text"], "Body"))
    if alg.get("steps"):
        for i, s in enumerate(alg["steps"], 1):
            fl.append(para(f"<b>{i}.</b>&nbsp; {s}", "Bullet"))
    if alg.get("math"):
        fl.append(math_block(alg["math"]))
    if alg.get("after"):
        fl.append(para(alg["after"], "Body"))

    fl.append(para("Assumptions &amp; when to use", "Sub"))
    fl.append(para(e["assumptions"], "Body"))

    if e.get("relationship"):
        fl.append(para("Relationship to other methods", "Sub"))
        fl.append(para(e["relationship"], "Body"))

    fl.append(para("References", "Sub"))
    for r in e["refs"]:
        fl.append(para(r, "Ref"))
    fl.append(rule(5, 7))
    return fl

def render_family(fam):
    fl = [anchor(fam["name"], 1), para(fam["name"], "Family"), rule(2, 4),
          para(fam["intro"], "FamilyIntro")]
    for e in fam["entries"]:
        e["family"] = fam["short"]
        fl += render_entry(e)
    return fl

print("renderers defined")

# ============================================================================
# CONTENT — grounded in papers/Quantification and papers/Quantification/Methods
# ============================================================================
FORMAN05 = "[1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate Classification. <i>ECML</i>, 564&ndash;575."
FORMAN06 = "[1] Forman, G. (2006). Quantifying Trends Accurately Despite Classifier Error and Class Imbalance. <i>KDD&#39;06</i>, 157&ndash;166."
FORMAN08 = "[2] Forman, G. (2008). Quantifying Counts and Costs via Classification. <i>Data Mining and Knowledge Discovery</i>, 17(2), 164&ndash;206."
BELLA10  = "[1] Bella, A., Ferri, C., Hern&aacute;ndez-Orallo, J., &amp; Ram&iacute;rez-Quintana, M. J. (2010). Quantification via Probability Estimators. <i>ICDM</i>, 737&ndash;742."
FIRAT16  = "[1] Firat, A. (2016). Unified Framework for Quantification. <i>arXiv:1606.00868</i>."
FRIED15  = "[1] Friedman, J. (2014). Class Counts in Future Unlabeled Samples (Detecting and Dealing with Concept Drift). <i>MIT CSAIL Big Data Event</i>."

counting = {
 "name": "Counting", "short": "Counting",
 "intro": "Counting methods estimate prevalence from the classifier&#39;s output on the test "
          "set and, except for the naive baselines, correct that count for the classifier&#39;s "
          "own error rates. They are the oldest and most widely used quantifier family.",
 "entries": [
  # ---- CC
  {"name": "CC", "title": "Classify &amp; Count",
   "imp": "from mlquantify.counting import CC",
   "summary": "Classify &amp; Count (CC) quantifier.",
   "description": "Targets prior probability shift. Classifies every test instance with a hard "
       "classifier and reports the fraction assigned to each class as the prevalence estimate. "
       "It is the simplest baseline and is systematically biased whenever the test class "
       "distribution differs from the training one.",
   "params": [("estimator", "A classifier with <font face='Courier'>fit</font>/<font face='Courier'>predict</font>. If <font face='Courier'>None</font>, call <font face='Courier'>aggregate</font> with pre-computed labels."),
              ("threshold", "float, default=0.5. Decision threshold applied to soft scores to form hard labels.")],
   "attrs": [("estimator_", "The fitted underlying classifier."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "CC does not correct for classifier bias, so its error grows roughly linearly as the "
            "test prevalence moves away from the training prevalence. Contrast with PCC (uses "
            "soft counts) and ACC (corrects for TPR/FPR).",
   "see_also": "PCC, ACC",
   "example": ["&gt;&gt;&gt; from mlquantify.counting import CC",
               "&gt;&gt;&gt; from sklearn.linear_model import LogisticRegression",
               "&gt;&gt;&gt; q = CC(LogisticRegression()).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)            # predict() path",
               "{0: 0.47, 1: 0.53}",
               "&gt;&gt;&gt; preds = q.estimator_.predict(X_test)",
               "&gt;&gt;&gt; q.aggregate(preds, y_train=y_train)   # aggregate() path",
               "{0: 0.47, 1: 0.53}"],
   "refs": [FORMAN05, FORMAN08],
   "problem": "Let h be a classifier trained on a labelled sample. Under prior probability shift "
       "the densities p(x|y) are shared by train and test while the priors p(y) differ. The goal "
       "is to estimate the test priors p<sub>te</sub>(y) from an unlabelled test bag.",
   "algorithm": {"text": "CC takes the empirical distribution of hard predictions as the estimate:",
       "math": ["p_hat(y) = (1/n) * sum_{x in test}  I( h(x) = y )"],
       "after": "No correction is applied: the raw classifier votes are simply counted and normalised."},
   "assumptions": "Unbiased only when the classifier is perfect or when the test distribution equals "
       "the training distribution. Otherwise it inherits the classifier&#39;s confusion and "
       "under/over-estimates the minority class. Use it only as a baseline or when the classifier "
       "is extremely accurate.",
   "relationship": "Root of the counting family. PCC replaces the hard vote I(h(x)=y) with the "
       "posterior probability; ACC post-corrects CC using estimated TPR and FPR.",
  },
  # ---- PCC
  {"name": "PCC", "title": "Probabilistic Classify &amp; Count",
   "imp": "from mlquantify.counting import PCC",
   "summary": "Probabilistic Classify &amp; Count (PCC) quantifier.",
   "description": "Targets prior probability shift. Averages the posterior probabilities returned "
       "by a probabilistic classifier over the test bag instead of counting hard labels. Generally "
       "less biased than CC but, lacking any correction, still drifts under strong shift and "
       "requires a calibrated classifier.",
   "params": [("estimator", "A classifier with <font face='Courier'>fit</font>/<font face='Courier'>predict_proba</font>. If <font face='Courier'>None</font>, call <font face='Courier'>aggregate</font> with posteriors.")],
   "attrs": [("estimator_", "The fitted underlying classifier."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "PCC is unbiased only if the posteriors are well calibrated on the test distribution; "
            "calibration drift re-introduces bias. It does not correct for classifier error.",
   "see_also": "CC, ACC, GPACC",
   "example": ["&gt;&gt;&gt; from mlquantify.counting import PCC",
               "&gt;&gt;&gt; q = PCC(LogisticRegression()).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.48, 1: 0.52}",
               "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
               "&gt;&gt;&gt; q.aggregate(proba)",
               "{0: 0.48, 1: 0.52}"],
   "refs": [BELLA10, FORMAN05.replace("[1]", "[2]")],
   "problem": "Same prior-shift setting as CC, but the classifier exposes posteriors "
       "P(y|x) = p_hat_te(y|x). PCC treats the mean posterior as the prevalence.",
   "algorithm": {"text": "Average the soft posteriors over the test bag:",
       "math": ["p_hat(y) = (1/n) * sum_{x in test}  P( y | x )"],
       "after": "Equivalent to CC when posteriors are hardened to 0/1, but soft counts reduce "
                "variance and exploit classifier confidence."},
   "assumptions": "Unbiased under prior shift only with calibrated posteriors; otherwise biased like "
       "CC. Prefer it over CC whenever reliable probabilities are available and the shift is mild.",
   "relationship": "Probabilistic counterpart of CC. GPACC is its multiclass, constrained-regression "
       "generalisation; ACC/PACC add an explicit error correction that PCC omits.",
  },
  # ---- ACC
  {"name": "ACC", "title": "Adjusted Classify &amp; Count",
   "imp": "from mlquantify.counting import ACC",
   "summary": "Adjusted Classify &amp; Count (ACC) quantifier.",
   "description": "Targets prior probability shift. Runs CC and then corrects the biased count using "
       "the classifier&#39;s true- and false-positive rates, estimated by cross-validation on the "
       "training set. The correction is exact in expectation but becomes unstable when the rates "
       "are close together (low-separability classifiers).",
   "params": [("estimator", "A classifier with <font face='Courier'>fit</font>/<font face='Courier'>predict</font>."),
              ("threshold", "float, default=0.5. Operating point at which TPR and FPR are measured."),
              ("cv", "int, cross-validation folds used to estimate TPR and FPR.")],
   "attrs": [("estimator_", "The fitted underlying classifier."),
             ("tpr_", "Estimated true-positive rate at the chosen threshold."),
             ("fpr_", "Estimated false-positive rate at the chosen threshold."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "ACC corrects the linear bias of CC and is unbiased when the estimated TPR/FPR match the "
            "test set. The estimate is clipped to [0,1]; the denominator (TPR&minus;FPR) controls "
            "its variance.",
   "see_also": "CC, TAC, TX, TMAX, T50, MS, MS2",
   "example": ["&gt;&gt;&gt; from mlquantify.counting import ACC",
               "&gt;&gt;&gt; q = ACC(LogisticRegression()).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.30, 1: 0.70}",
               "&gt;&gt;&gt; preds = q.estimator_.predict(X_test)",
               "&gt;&gt;&gt; q.aggregate(preds, y_train=y_train)",
               "{0: 0.30, 1: 0.70}"],
   "refs": [FORMAN05, FORMAN08],
   "problem": "Binary prior shift. The observed positive count produced by the classifier mixes true "
       "positives and false positives at the (shift-invariant) rates TPR and FPR.",
   "algorithm": {"text": "The observed positive rate decomposes as a linear function of the true "
       "prevalence p:",
       "math": ["CC = TPR * p + FPR * (1 - p)",
                "    = (TPR - FPR) * p + FPR",
                "=>  p_hat = ( CC - FPR ) / ( TPR - FPR ) ,  clipped to [0, 1]"],
       "after": "TPR and FPR are estimated once by cross-validation on the training data and reused "
                "for any test bag. Multiclass uses the analogous misclassification-matrix inversion."},
   "assumptions": "Unbiased when TPR/FPR are stable across the shift (prior-shift assumption). When "
       "TPR&minus;FPR is small &mdash; a weak classifier under heavy imbalance &mdash; the division "
       "amplifies estimation error, so ACC trades CC&#39;s bias for variance. Best with a reasonably "
       "separating classifier.",
   "relationship": "Adds the TPR/FPR correction missing from CC. The threshold-policy methods (TAC, "
       "TX, TMAX, T50, MS, MS2) are ACC evaluated at cleverly chosen operating points.",
  },
 ],
}
print("counting defined")

def _thr(name, title, policy_desc, policy_text, policy_math, note, refs):
    return {
     "name": name, "title": title,
     "imp": f"from mlquantify.counting import {name}",
     "summary": f"{title} ({name}) quantifier.",
     "description": "Targets prior probability shift. A threshold-policy member of the Adjusted Count "
        "family: it selects a classification threshold by a fixed rule, measures TPR and FPR there "
        f"by cross-validation, and applies the adjusted-count correction. {policy_desc} Binary-only.",
     "params": [("estimator", "A classifier exposing soft scores (<font face='Courier'>predict_proba</font>/<font face='Courier'>decision_function</font>)."),
                ("cv", "int, cross-validation folds used to build the threshold&ndash;rate table.")],
     "attrs": [("estimator_", "The fitted underlying classifier."),
               ("threshold_", "The operating threshold chosen by the policy."),
               ("tpr_", "True-positive rate at the chosen threshold(s)."),
               ("fpr_", "False-positive rate at the chosen threshold(s)."),
               ("classes_", "Class labels seen during fit.")],
     "notes": note,
     "see_also": "ACC, TAC, TX, TMAX, T50, MS, MS2",
     "example": [f"&gt;&gt;&gt; from mlquantify.counting import {name}",
                 f"&gt;&gt;&gt; q = {name}(LogisticRegression()).fit(X_train, y_train)",
                 "&gt;&gt;&gt; q.predict(X_test)",
                 "{0: 0.41, 1: 0.59}",
                 "&gt;&gt;&gt; scores = q.estimator_.predict_proba(X_test)[:, 1]",
                 "&gt;&gt;&gt; q.aggregate(scores, y_train=y_train)",
                 "{0: 0.41, 1: 0.59}"],
     "refs": refs,
     "problem": "Binary prior shift. ACC&#39;s correction is exact only if TPR&minus;FPR is well away "
        "from zero; the choice of operating threshold therefore controls the estimator&#39;s "
        "stability. Each policy picks that threshold differently from the cross-validated "
        "TPR(t)/FPR(t) curves.",
     "algorithm": {"text": policy_text,
        "math": policy_math,
        "after": "With the threshold fixed, the standard adjusted count "
                 "p_hat = (CC &minus; FPR)/(TPR &minus; FPR) is applied at that point."},
     "assumptions": "Same prior-shift assumption as ACC. The policy aims to keep TPR&minus;FPR large "
        "(reliable denominator). Threshold rules that pin a specific rate (T50, TX) are robust under "
        "imbalance; TMAX maximises separation but can carry a systematic bias.",
     "relationship": "All are ACC at a chosen operating point. MS/MS2 aggregate over many thresholds "
        "instead of committing to one, trading a little bias for much lower variance.",
    }

counting["entries"].extend([
 _thr("TAC", "Threshold Adjusted Count",
   "TAC applies the correction at the default decision threshold.",
   "TAC fixes the operating point at the classifier&#39;s default threshold (0.5) and corrects there:",
   ["t* = 0.5", "p_hat = ( CC(t*) - FPR(t*) ) / ( TPR(t*) - FPR(t*) )"],
   "Equivalent to binary ACC at threshold 0.5; the simplest threshold policy.",
   [FORMAN06, FORMAN08]),
 _thr("TX", "Threshold at the X-crossing",
   "TX (the &#39;X method&#39;) picks the threshold where FPR(t) crosses 1&minus;TPR(t).",
   "TX chooses the operating point where the false-positive rate equals the false-negative rate, "
   "i.e. where the two error curves cross:",
   ["t* = argmin_t | FPR(t) - (1 - TPR(t)) |", "   (equivalently  TPR(t*) = 1 - FPR(t*) )"],
   "The crossing point keeps both error rates moderate and TPR&minus;FPR comfortably non-zero, "
   "giving a stable denominator under imbalance.",
   [FORMAN06, FORMAN08]),
 _thr("TMAX", "Threshold at maximum TPR&minus;FPR",
   "TMAX (the &#39;Max method&#39;) maximises the gap TPR(t)&minus;FPR(t).",
   "TMAX selects the threshold that maximises classifier separation (the Youden index); ties are "
   "resolved at the midpoint of the maximal range:",
   ["t* = argmax_t ( TPR(t) - FPR(t) )"],
   "Maximising TPR&minus;FPR yields the most stable denominator, but Forman reports TMAX carries a "
   "systematic linear bias because the chosen rates often fail to transfer to the test set.",
   [FORMAN06, FORMAN08]),
 _thr("T50", "Threshold at TPR = 0.5",
   "T50 pins the threshold where the true-positive rate equals 0.5.",
   "T50 chooses the operating point at which exactly half of the positives are recovered:",
   ["t* = argmin_t | TPR(t) - 0.5 |"],
   "Fixing TPR makes the policy insensitive to positive-class scarcity; robust with as few as ~10 "
   "training positives, per Forman.",
   [FORMAN06, FORMAN08]),
 _thr("MS", "Median Sweep",
   "MS computes an adjusted count at every threshold and returns the median.",
   "Rather than trusting a single operating point, MS sweeps all thresholds, computes one adjusted "
   "count per threshold, and reports the median estimate:",
   ["for each threshold t:  p_t = ( CC(t) - FPR(t) ) / ( TPR(t) - FPR(t) )",
    "p_hat = median_t  p_t"],
   "The median is robust to the unreliable thresholds (small TPR&minus;FPR) that would wreck a "
   "single-point ACC; Forman found Median Sweep outstandingly robust across shifts.",
   [FORMAN06, FORMAN08]),
 _thr("MS2", "Median Sweep (restricted)",
   "MS2 is Median Sweep restricted to thresholds with a reliable denominator.",
   "MS2 runs the same sweep but discards operating points where TPR(t)&minus;FPR(t) is too small "
   "(below 0.25) before taking the median:",
   ["S = { t : ( TPR(t) - FPR(t) ) > 0.25 }",
    "p_hat = median_{t in S}  ( CC(t) - FPR(t) ) / ( TPR(t) - FPR(t) )"],
   "By pruning low-separation thresholds MS2 removes the highest-variance terms of the sweep, "
   "usually improving on MS when the classifier is weak in part of the score range.",
   [FORMAN06, FORMAN08]),
])

# ---- Generalized counting (FM, GACC, GPACC): the Firat unified framework
def _gen(name, title, transform_desc, transform, loss_name, loss_math, soft, refs, extra=None):
    return {
     "name": name, "title": title,
     "imp": f"from mlquantify.counting import {name}",
     "summary": f"{title} ({name}) quantifier.",
     "description": "Targets prior probability shift. A multiclass quantifier built on the unified "
        "constrained-regression framework: it summarises train and test through a feature transform "
        f"and solves for the prevalence vector on the simplex. {transform_desc} "
        + ("Requires soft posteriors. " if soft else "")
        + "Native multiclass (no One-vs-All).",
     "params": [("estimator", "A probabilistic classifier" + (" (uses <font face='Courier'>predict_proba</font>)." if soft else ".")),
                ("solver", "Simplex optimiser, default <font face='Courier'>'slsqp'</font> (see solvers)."),
                ("cv", "int, cross-validation folds used to build the per-class transform matrix.")],
     "attrs": [("estimator_", "The fitted underlying classifier."),
               ("representation_", "The fitted feature transform (per-class matrix X)."),
               ("classes_", "Class labels seen during fit.")],
     "notes": (extra or "") + " Solved on the probability simplex (p &ge; 0, sum p = 1), so the "
              "estimate is always a valid distribution.",
     "see_also": "GACC, GPACC, FM, GHDy, GHDx",
     "example": [f"&gt;&gt;&gt; from mlquantify.counting import {name}",
                 f"&gt;&gt;&gt; q = {name}(LogisticRegression()).fit(X_train, y_train)",
                 "&gt;&gt;&gt; q.predict(X_test)            # multiclass",
                 "{0: 0.2, 1: 0.5, 2: 0.3}",
                 "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
                 "&gt;&gt;&gt; q.aggregate(proba)",
                 "{0: 0.2, 1: 0.5, 2: 0.3}"],
     "refs": refs,
     "problem": "Firat (2016) unifies quantifiers as a constrained multivariate regression. With a "
        "feature transform f(x), let X be the matrix whose column k is the per-class mean transform "
        "and y the mean transform on the test bag. Under prior shift the test transform is the "
        "prevalence-weighted mixture of the per-class transforms:",
     "algorithm": {"text": f"{title} uses the transform: {transform}. The prevalence solves the "
        "constrained regression",
        "math": ["minimize_p   L( y ,  X p )",
                 "subject to   p >= 0 ,   sum_k p_k = 1",
                 ""] + loss_math,
        "after": "X is estimated once by cross-validation on the training set; the same X serves any "
                 "test bag. Because the model is multivariate, the binary method extends to "
                 "multiclass automatically."},
     "assumptions": "Requires that the chosen transform&#39;s class-conditional distribution is "
        "invariant across the shift (an extension of prior-shift to transform space) and that the "
        "per-class transforms are sufficiently distinct (identifiability). Best for multiclass "
        "problems where One-vs-All counting breaks down.",
     "relationship": "Members of one framework differing only in transform and loss: "
        "GACC uses hard one-hot counts, GPACC uses soft posteriors, FM uses Friedman&#39;s "
        "prior-threshold indicator, GHDy/GHDx use histograms with a Hellinger loss.",
    }

counting["entries"].extend([
 _gen("GACC", "Generalized Adjusted Count",
   "GACC transforms each instance into its hard one-hot predicted class and matches the resulting "
   "class-frequency vectors.",
   "f(x) = onehot(argmax_k P(k|x))  (hard class assignment)",
   "least squares", ["L( y, Xp ) = || y - X p ||^2     (least-squares / L2)"],
   False, [FIRAT16, FORMAN08],
   "Multiclass generalisation of ACC: the per-class matrix X is the training confusion matrix, and "
   "solving Xp=y inverts it on the simplex."),
 _gen("GPACC", "Generalized Probabilistic Adjusted Count",
   "GPACC transforms each instance into its soft posterior vector and matches the mean posteriors.",
   "f(x) = P(.|x)  (soft posterior vector)",
   "least squares", ["L( y, Xp ) = || y - X p ||^2     (least-squares / L2)"],
   True, [FIRAT16, BELLA10],
   "Multiclass, corrected generalisation of PCC (a.k.a. PACC): the soft-count matrix is inverted on "
   "the simplex instead of being read off directly."),
 _gen("FM", "Friedman&#39;s Method",
   "FM transforms each instance with an indicator of whether its posterior for class k exceeds the "
   "training prior of class k.",
   "f_k(x) = I( P(k|x) >= prior_tr(k) )  (Friedman threshold = training prior)",
   "least squares", ["L( y, Xp ) = || y - X p ||^2     (least-squares / L2)"],
   True, [FRIED15, FIRAT16],
   "Friedman&#39;s dynamic-threshold quantifier: thresholding the posterior at the class&#39;s own "
   "training prior minimises the variance of the proportion estimate."),
])
print("counting complete:", len(counting["entries"]), "entries")

# ============================================================================
GC13   = "[1] Gonz&aacute;lez-Castro, V., Alaiz-Rodr&iacute;guez, R., &amp; Alegre, E. (2013). Class Distribution Estimation Based on the Hellinger Distance. <i>Information Sciences</i>, 218, 146&ndash;164."
DYS19  = "[1] Maletzke, A., dos Reis, D., Cherman, E., &amp; Batista, G. (2019). DyS: A Framework for Mixture Models in Quantification. <i>AAAI</i>, 33, 4552&ndash;4560."
SCORE21= "[1] Maletzke, A., dos Reis, D., Cherman, E., &amp; Batista, G. (2019). DyS: A Framework for Mixture Models in Quantification. <i>AAAI</i>, 33, 4552&ndash;4560."
IYER14 = "[1] Iyer, A., Nath, S., &amp; Sarawagi, S. (2014). Maximum Mean Discrepancy for Class Ratio Estimation: Convergence Bounds and Kernel Selection. <i>ICML</i>, 32."
MOREO24= "[1] Moreo, A., Gonz&aacute;lez, P., &amp; del Coz, J. J. (2024). Kernel Density Estimation for Multiclass Quantification. <i>arXiv:2401.00490</i>."
DELCOZ22="[1] del Coz, J. J. (2022). UniOviedo(Team2) at LeQua 2022: Comparison of Traditional Quantifiers and a New Method Based on Energy Distance. <i>CLEF 2022</i>."
KAWA16 = "[2] Kawakubo, H., du Plessis, M. C., &amp; Sugiyama, M. (2016). Computationally Efficient Class-Prior Estimation under Class Balance Change Using Energy Distance. <i>IEICE Trans.</i>, 99(1), 176&ndash;186."

matching = {
 "name": "Matching", "short": "Matching",
 "intro": "Distribution-matching (mixture-model) methods model the test sample as a "
          "prevalence-weighted mixture of the per-class distributions and search for the mixing "
          "weights that make the mixture most similar to the observed test distribution. They "
          "differ in how a distribution is represented (histogram, kernel, raw sample, embedding) "
          "and in the divergence minimised.",
 "entries": [
  # ---- DyS
  {"name": "DyS", "title": "Distribution y-Similarity",
   "imp": "from mlquantify.matching import DyS",
   "summary": "Distribution y-Similarity (DyS) quantifier.",
   "description": "Targets prior probability shift. A general mixture-model framework: it builds "
       "score histograms for each class, mixes them by a candidate prevalence, and searches for the "
       "mixture that minimises a chosen histogram dissimilarity against the test histogram. HDy is "
       "a special case; Topsoe is the recommended distance. Binary base method.",
   "params": [("estimator", "A probabilistic classifier (positive-class scores)."),
              ("measure", ("Histogram dissimilarity minimised between the mixed and test histograms; "
                  "default <font face='Courier'>'topsoe'</font>.", [
                  ("topsoe", "symmetric information-theoretic distance; most accurate when bins are tuned (recommended)."),
                  ("hellinger", "bounded distance over sqrt-probabilities; the classic HDy choice."),
                  ("prob_symm", "probabilistic symmetric chi-square distance."),
                  ("sqEuclidean", "squared Euclidean distance between the bin vectors.")])),
              ("bins_size", "array of bin counts to sweep; controls histogram resolution. Smaller "
                  "counts (&le; ~20) usually work best; estimates are aggregated by their median."),
              ("solver", ("Scalar search strategy over the prevalence alpha.", [
                  ("ternary", "trisection search; fast, exploits the near-unimodal objective (default)."),
                  ("grid", "exhaustive search over an evenly-spaced grid of alpha values."),
                  ("bounded", "scipy bounded scalar minimiser.")]))],
   "attrs": [("estimator_", "The fitted underlying classifier."),
             ("pos_scores_", "Cross-validated positive-class score sample."),
             ("neg_scores_", "Cross-validated negative-class score sample."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "Best bin counts are small (typically &le; 20), contradicting HDy&#39;s original 10&ndash;110 "
            "range; the per-bin estimates are aggregated by their median. Ternary search suffices "
            "because the objective is near-unimodal in alpha.",
   "see_also": "HDy, SORD, SMM, HDx",
   "example": ["&gt;&gt;&gt; from mlquantify.matching import DyS",
               "&gt;&gt;&gt; q = DyS(LogisticRegression(), measure='topsoe').fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.49, 1: 0.51}",
               "&gt;&gt;&gt; ts = q.estimator_.predict_proba(X_test)[:, 1]",
               "&gt;&gt;&gt; q.aggregate(ts)",
               "{0: 0.48, 1: 0.52}"],
   "refs": [DYS19, GC13.replace("[1]", "[2]")],
   "problem": "Binary prior shift. Let H(S) be the score histogram of a sample S. The positive "
       "training scores S+ and negative scores S&minus; give class histograms; the test scores Z "
       "give the observed histogram. The test histogram is assumed to be the alpha-mixture of the "
       "class histograms, alpha being the positive prevalence.",
   "algorithm": {"text": "Search the mixing weight alpha that minimises a dissimilarity DS between "
       "the mixed class histogram and the test histogram:",
       "math": ["alpha* = argmin_{0 <= alpha <= 1}",
                "         DS( alpha*H(S+) + (1-alpha)*H(S-) ,  H(Z) )",
                "p_hat = [ 1 - alpha* , alpha* ]"],
       "after": "The search uses ternary search; the procedure is repeated for several bin counts "
                "and the median alpha is returned. With DS = Hellinger this reduces exactly to HDy."},
   "assumptions": "Needs scores whose class-conditional distribution is shift-invariant, and a test "
       "bag large enough to estimate H(Z). Robust under imbalance; the parameter-sensitive choice is "
       "the bin count, which is why SORD/SMM (bin-free members) exist. Strong general-purpose binary "
       "quantifier.",
   "relationship": "Generalises HDy (Hellinger) to any symmetric distance. SORD and SMM are DyS "
       "members that drop histograms (operate on the raw score sample and the score mean).",
  },
  # ---- HDy
  {"name": "HDy", "title": "Hellinger Distance y",
   "imp": "from mlquantify.matching import HDy",
   "summary": "Hellinger Distance y (HDy) quantifier.",
   "description": "Targets prior probability shift. Represents each class by a histogram of classifier "
       "scores and finds the positive prevalence whose mixed histogram minimises the Hellinger "
       "distance to the test histogram. The original mixture-model quantifier on posteriors; "
       "binary base method run over a range of bin counts.",
   "params": [("estimator", "A probabilistic classifier (positive-class scores)."),
              ("bins_size", "array of bin counts to sweep; controls histogram resolution. Estimates "
                  "over the different bin counts are aggregated by their median."),
              ("solver", ("Scalar search strategy over the positive prevalence alpha.", [
                  ("linear", "scan alpha on an evenly-spaced grid; the original HDy search."),
                  ("ternary", "interval-trisection search; faster and usually as precise.")]))],
   "attrs": [("estimator_", "The fitted underlying classifier."),
             ("pos_scores_", "Cross-validated positive-class score sample."),
             ("neg_scores_", "Cross-validated negative-class score sample."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "Estimates are computed for several bin counts and aggregated by the median. HDy is the "
            "Hellinger instance of DyS; switching to Topsoe (DyS) often lowers error.",
   "see_also": "DyS, HDx, GHDy",
   "example": ["&gt;&gt;&gt; from mlquantify.matching import HDy",
               "&gt;&gt;&gt; q = HDy(LogisticRegression()).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.49, 1: 0.51}",
               "&gt;&gt;&gt; ts = q.estimator_.predict_proba(X_test)[:, 1]",
               "&gt;&gt;&gt; q.aggregate(ts)",
               "{0: 0.49, 1: 0.51}"],
   "refs": [GC13, DYS19.replace("[1]", "[2]")],
   "problem": "Binary prior shift with class score histograms P+, P&minus; and test histogram Q, "
       "each a normalised vector over b bins.",
   "algorithm": {"text": "Minimise the Hellinger distance between the mixed histogram and the test "
       "histogram over the positive prevalence alpha:",
       "math": ["HD(P, Q) = sqrt( 1 - sum_i sqrt( P_i * Q_i ) )",
                "alpha* = argmin_{0<=alpha<=1}",
                "         HD( alpha*P+ + (1-alpha)*P- ,  Q )"],
       "after": "Repeated for bins 10,20,...,110 in the original paper; the final estimate is the "
                "median over bin counts. mlquantify also offers Laplace-smoothed histograms."},
   "assumptions": "Same prior-shift / shift-invariant-score assumption as DyS. Sensitive to the bin "
       "count when scores are sparse; the median over bins mitigates this. A robust, well-tested "
       "binary default.",
   "relationship": "The canonical mixture model; DyS generalises its distance, HDx moves the matching "
       "into feature space, and GHDy lifts it to multiclass constrained regression.",
  },
  # ---- HDx
  {"name": "HDx", "title": "Hellinger Distance x",
   "imp": "from mlquantify.matching import HDx",
   "summary": "Hellinger Distance x (HDx) quantifier.",
   "description": "Targets prior probability shift. The classifier-free mixture model: it builds a "
       "histogram for every input feature and matches the prevalence-mixed per-feature histograms "
       "to the test histograms under the Hellinger distance. Needs no scorer, only the raw features. "
       "Binary base method.",
   "params": [("bins_size", "array of bin counts to sweep per feature."),
              ("solver", "Scalar search over alpha.")],
   "attrs": [("pos_repr_", "Per-feature histograms of the positive class."),
             ("neg_repr_", "Per-feature histograms of the negative class."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "Because it skips the classifier, HDx avoids calibration issues but cannot exploit a "
            "good scorer; per-feature distances are aggregated (median) into one estimate.",
   "see_also": "HDy, GHDx",
   "example": ["&gt;&gt;&gt; from mlquantify.matching import HDx",
               "&gt;&gt;&gt; q = HDx().fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.52, 1: 0.48}",
               "&gt;&gt;&gt; q.aggregate(X_test)      # features, no classifier",
               "{0: 0.52, 1: 0.48}"],
   "refs": [GC13],
   "problem": "Binary prior shift matched directly in feature space: each feature j has class "
       "histograms P+^j, P&minus;^j and test histogram Q^j.",
   "algorithm": {"text": "For each feature find the alpha minimising the Hellinger distance, then "
       "aggregate across features:",
       "math": ["alpha_j* = argmin_alpha  HD( alpha*P+^j + (1-alpha)*P-^j ,  Q^j )",
                "p_hat(+) = median_j  alpha_j*"],
       "after": "Sweeping bin counts and taking medians (over bins and features) yields the final "
                "estimate. No cross-validation is needed since there is no classifier."},
   "assumptions": "Assumes each feature&#39;s class-conditional distribution is shift-invariant and "
       "informative. Works without a trained model, useful when a reliable classifier is "
       "unavailable, but degrades with many weak/correlated features.",
   "relationship": "Feature-space twin of HDy. GHDx is its multiclass constrained-regression form.",
  },
 ],
}
print("matching part 1 defined")

matching["entries"].extend([
 # ---- SORD
 {"name": "SORD", "title": "Sample Ordinal Distance",
  "imp": "from mlquantify.matching import SORD",
  "summary": "Sample Ordinal Distance (SORD) quantifier.",
  "description": "Targets prior probability shift. A parameter-free mixture model: instead of "
      "binning scores it compares the raw weighted score samples with an ordinal (earth-mover-like) "
      "distance, searching the prevalence that best mixes the positive and negative score samples "
      "to match the test sample. Binary base method, no bins to tune.",
  "params": [("estimator", "A probabilistic classifier (positive-class scores)."),
             ("solver", "Scalar search over alpha (ternary search).")],
  "attrs": [("estimator_", "The fitted underlying classifier."),
            ("pos_scores_", "Cross-validated positive-class score sample."),
            ("neg_scores_", "Cross-validated negative-class score sample."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Parameter-free (no bin count). Cost grows as O(n log n) in the combined sample size, so "
           "training scores are sub-sampled (~1000/class). Competitive with a tuned Topsoe-DyS.",
  "see_also": "DyS, SMM, HDy",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import SORD",
              "&gt;&gt;&gt; q = SORD(LogisticRegression()).fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.50, 1: 0.50}",
              "&gt;&gt;&gt; ts = q.estimator_.predict_proba(X_test)[:, 1]",
              "&gt;&gt;&gt; q.aggregate(ts)",
              "{0: 0.50, 1: 0.50}"],
  "refs": [SCORE21],
  "problem": "Binary prior shift compared on raw score samples. Each observation is given unit "
      "mass scaled by its sample size so the mixed training sample and the test sample carry equal "
      "total mass; SORD is the minimum cost to morph one weighted sample into the other (a 1-D "
      "earth-mover / MDPA distance, the bins-to-infinity limit of ordinal distance).",
  "algorithm": {"text": "Assign signed weights to the pooled, sorted scores and accumulate the "
      "absolute running mass times the gaps:",
      "math": ["w(x) = +alpha/|S+|     if x in S+",
               "       +(1-alpha)/|S-| if x in S-",
               "       -1/|Z|          if x in Z   (test)",
               "v = sort( S+  U  S-  U  Z )",
               "SORD(alpha) = sum_i | (v_i - v_{i-1}) * cumsum(w)_i |",
               "alpha* = argmin_alpha  SORD(alpha)"],
      "after": "Ternary search over alpha; no histogram and therefore no bin parameter."},
  "assumptions": "Same shift-invariant-score assumption as DyS, but immune to bin mis-specification "
      "and the curse of dimensionality in score space. Use it when you want a robust, tuning-free "
      "binary quantifier and can afford the slightly higher compute.",
  "relationship": "A bin-free member of the DyS framework (H(x)=x). Where SMM matches only the mean "
      "of the score sample, SORD matches the whole ordinal shape.",
 },
 # ---- SMM
 {"name": "SMM", "title": "Sample Mean Matching",
  "imp": "from mlquantify.matching import SMM",
  "summary": "Sample Mean Matching (SMM) quantifier.",
  "description": "Targets prior probability shift. The lightest mixture model: it summarises each "
      "class by its mean classifier score and solves in closed form for the prevalence that makes "
      "the mixed mean equal the test mean score. Binary base method, no search and no bins.",
  "params": [("estimator", "A probabilistic classifier (positive-class scores).")],
  "attrs": [("estimator_", "The fitted underlying classifier."),
            ("pos_mean_", "Mean positive-class score (cross-validated)."),
            ("neg_mean_", "Mean negative-class score (cross-validated)."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Closed-form and extremely fast, but uses only the first moment of the score "
           "distribution, so it is less accurate than DyS/SORD when class scores overlap.",
  "see_also": "DyS, SORD",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import SMM",
              "&gt;&gt;&gt; q = SMM(LogisticRegression()).fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.46, 1: 0.54}",
              "&gt;&gt;&gt; ts = q.estimator_.predict_proba(X_test)[:, 1]",
              "&gt;&gt;&gt; q.aggregate(ts)",
              "{0: 0.46, 1: 0.54}"],
  "refs": [SCORE21],
  "problem": "Binary prior shift matched on the mean score. With mean positive score mu+, mean "
      "negative score mu&minus; and mean test score mu_Z, the test mean is the alpha-mixture of the "
      "class means.",
  "algorithm": {"text": "Solve the one-equation mean-matching condition directly:",
      "math": ["alpha * mu+ + (1 - alpha) * mu- = mu_Z",
               "alpha* = ( mu_Z - mu- ) / ( mu+ - mu- )   (clipped to [0,1])"],
      "after": "No optimisation loop; a single division gives the estimate."},
  "assumptions": "Exact only if the class score distributions differ enough in mean and are "
      "shift-invariant. Degenerates when mu+ &asymp; mu&minus; (overlapping classes). Best as a fast "
      "baseline or when scores are well separated.",
  "relationship": "The first-moment special case of DyS/SORD: matching means instead of full "
      "histograms or ordinal shapes.",
 },
 # ---- MMD_RKHS
 {"name": "MMD_RKHS", "title": "Maximum Mean Discrepancy (RKHS)",
  "imp": "from mlquantify.matching import MMD_RKHS",
  "summary": "Maximum Mean Discrepancy in an RKHS (MMD_RKHS) quantifier.",
  "description": "Targets prior probability shift. Matches distributions by their mean embeddings in "
      "a reproducing-kernel Hilbert space: it finds the prevalence vector whose mixture of per-class "
      "mean embeddings is closest to the test mean embedding. Kernel-based, multiclass, provably "
      "consistent, and free of density or histogram estimation.",
  "params": [("kernel", ("Kernel defining the RKHS feature map; must be universal for "
                 "consistency. Default <font face='Courier'>'rbf'</font>.", [
                 ("rbf", "Gaussian radial-basis kernel; universal, the default choice."),
                 ("linear", "plain inner product; fast but only matches first moments."),
                 ("poly", "polynomial kernel of a given degree; matches higher moments.")])),
             ("gamma", "Kernel bandwidth (RBF/poly). Small gamma over-smooths; large gamma "
                 "over-fits the embedding. <font face='Courier'>None</font> uses 1/n_features."),
             ("solver", "Simplex optimiser for the convex QP, default <font face='Courier'>'slsqp'</font>.")],
  "attrs": [("representation_", "Fitted kernel-mean embeddings per class."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Statistically consistent under a universal kernel; its error shrinks when classes are "
           "well separated (large minimum eigenvalue) and the data spread is small. Kernel choice "
           "matters and can be tuned.",
  "see_also": "EDy, KDEyML",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import MMD_RKHS",
              "&gt;&gt;&gt; q = MMD_RKHS(kernel='rbf').fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.33, 1: 0.33, 2: 0.34}",
              "&gt;&gt;&gt; q.aggregate(X_test)",
              "{0: 0.33, 1: 0.33, 2: 0.34}"],
  "refs": [IYER14],
  "problem": "Prior shift with a universal kernel K and feature map Phi. The test distribution is "
      "the prevalence-weighted mixture of the class-conditional distributions; matching is done "
      "between their mean embeddings Phi_y = E[Phi(x)|y] and the test mean Phi_U.",
  "algorithm": {"text": "Minimise the squared RKHS distance between the mixed and test mean "
      "embeddings &mdash; a convex quadratic program on the simplex:",
      "math": ["minimize_theta  || sum_y theta_y * Phi_y  -  Phi_U ||^2",
               "subject to      theta >= 0 ,  sum_y theta_y = 1",
               "kernelised:     theta^T (A^T A) theta - 2 theta^T (A^T a)",
               "  A = [Phi_1-Phi_0, ...] ,  a = Phi_U - Phi_0  (inner products via K)"],
      "after": "All inner products are computed with the kernel trick; the QP is solved on the "
               "simplex. Estimator-free &mdash; it embeds features directly."},
  "assumptions": "Consistent when K is universal and the class mixtures are identifiable "
      "(distinct embeddings). Needs enough test points to estimate Phi_U. Strong choice for "
      "multiclass and high-dimensional data; sensitive to kernel bandwidth.",
  "relationship": "Like EDy/EDx it matches whole samples rather than histograms, but in an RKHS via "
      "kernel mean embeddings; EDy uses an energy distance instead of a kernel norm.",
 },
])
print("matching part 2 defined")

def _kdey(name, title, variant_desc, obj_text, obj_math, note):
    return {
     "name": name, "title": title,
     "imp": f"from mlquantify.matching import {name}",
     "summary": f"{title} ({name}) quantifier.",
     "description": "Targets prior probability shift. A multiclass distribution-matching method that "
        "replaces the per-class histograms of HDy with a single multivariate kernel density estimate "
        f"over the posterior vectors on the simplex. {variant_desc} Native multiclass; scales to "
        "many classes where histogram matching fragments.",
     "params": [("estimator", "A probabilistic classifier (posterior vectors)."),
                ("bandwidth", "float, KDE bandwidth; controls density smoothness."),
                ("solver", "Simplex optimiser / search used to minimise the objective.")],
     "attrs": [("estimator_", "The fitted underlying classifier."),
               ("representation_", "Per-class fitted KDE models."),
               ("classes_", "Class labels seen during fit.")],
     "notes": note,
     "see_also": "KDEyML, KDEyHD, KDEyCS, GKDEyML",
     "example": [f"&gt;&gt;&gt; from mlquantify.matching import {name}",
                 f"&gt;&gt;&gt; q = {name}(LogisticRegression(), bandwidth=0.1).fit(X_train, y_train)",
                 "&gt;&gt;&gt; q.predict(X_test)",
                 "{0: 0.2, 1: 0.5, 2: 0.3}",
                 "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
                 "&gt;&gt;&gt; q.aggregate(proba)",
                 "{0: 0.2, 1: 0.5, 2: 0.3}"],
     "refs": [MOREO24],
     "problem": "Prior shift on the (n&minus;1)-simplex of posterior vectors. Each class i is "
        "modelled by a density p_i fitted with KDE; the test posteriors are assumed drawn from the "
        "prevalence mixture sum_i p_i_prev * p_i. KDE replaces histograms so the multivariate "
        "simplex is not fragmented into bins.",
     "algorithm": {"text": obj_text,
        "math": obj_math,
        "after": "Per-class KDEs are fitted once at training time; only the objective changes "
                 "between the ML, HD and CS variants."},
     "assumptions": "Assumes posteriors are shift-invariant per class and that the bandwidth is "
        "well chosen (over-smoothing flattens class differences). Excels at multiclass quantification "
        "with moderate-to-many classes; KDEyML benefits from recalibrated posteriors.",
     "relationship": "The KDE generalisation of histogram matching (HDy/DyS). KDEyML is a likelihood "
        "method (compare EMQ); KDEyHD/KDEyCS are distribution-matching with Hellinger / "
        "Cauchy-Schwarz divergences.",
    }

matching["entries"].extend([
 _kdey("KDEyML", "Kernel Density Estimation y &mdash; Maximum Likelihood",
   "KDEyML chooses the prevalence by maximum likelihood of the test posteriors under the mixture "
   "of per-class KDEs.",
   "Maximise the log-likelihood of the test posteriors under the KDE mixture:",
   ["maximize_p   sum_{x in test}  log( sum_i p_i * KDE_i(x) )",
    "subject to   p >= 0 ,  sum_i p_i = 1"],
   "Likelihood objective; often beats EM (EMQ) when posteriors are recalibrated. Equivalent in "
   "spirit to EMQ but with KDE-smoothed class densities."),
 _kdey("KDEyHD", "Kernel Density Estimation y &mdash; Hellinger",
   "KDEyHD minimises the Hellinger distance between the test KDE and the mixture KDE, estimated by "
   "Monte-Carlo sampling.",
   "Minimise the (Monte-Carlo estimated) Hellinger divergence between the mixture density and the "
   "test density:",
   ["minimize_p   HD( sum_i p_i * KDE_i ,  KDE_test )",
    "  HD^2(f,g) = 1 - integral sqrt( f * g )   (Monte-Carlo over simplex samples)",
    "subject to   p >= 0 ,  sum_i p_i = 1"],
   "Distribution-matching analogue of HDy lifted to multivariate KDEs; the integral is approximated "
   "by sampling points from the fitted mixture."),
 _kdey("KDEyCS", "Kernel Density Estimation y &mdash; Cauchy-Schwarz",
   "KDEyCS minimises the Cauchy-Schwarz divergence, which has a closed form for Gaussian KDEs.",
   "Minimise the Cauchy-Schwarz divergence, available in closed form (train-train kernel matrices "
   "are pre-computed):",
   ["minimize_p   D_CS( sum_i p_i*KDE_i ,  KDE_test )",
    "  D_CS(f,g) = -log( <f,g> / sqrt(<f,f> <g,g>) )",
    "  closed form ~  r^T B r   with B precomputed at fit time",
    "subject to   p >= 0 ,  sum_i p_i = 1"],
   "Closed-form and the most efficient KDEy variant: the expensive train-train Gaussian integrals "
   "B are computed once at training time."),
])
print("KDEy defined")

matching["entries"].extend([
 # ---- GHDy
 {"name": "GHDy", "title": "Generalized Hellinger Distance y",
  "imp": "from mlquantify.matching import GHDy",
  "summary": "Generalized Hellinger Distance y (GHDy) quantifier.",
  "description": "Targets prior probability shift. The multiclass, constrained-regression form of "
      "HDy: it builds per-class probability-mass histograms of the posteriors and solves on the "
      "simplex for the prevalence that matches the test histogram under a Hellinger objective. "
      "Native multiclass, requires soft posteriors.",
  "params": [("estimator", "A probabilistic classifier (posteriors)."),
             ("bins", "histogram bin count(s) over the posterior scores."),
             ("solver", "Simplex optimiser, default <font face='Courier'>'slsqp'</font>.")],
  "attrs": [("estimator_", "The fitted underlying classifier."),
            ("representation_", "Per-class posterior histograms (PMF)."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Uses the Hellinger-surrogate loss so it can be optimised with gradient-free simplex "
           "solvers; the binary HDy is recovered for two classes.",
  "see_also": "HDy, GHDx, GACC",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import GHDy",
              "&gt;&gt;&gt; q = GHDy(LogisticRegression()).fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.2, 1: 0.5, 2: 0.3}",
              "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
              "&gt;&gt;&gt; q.aggregate(proba)",
              "{0: 0.2, 1: 0.5, 2: 0.3}"],
  "refs": [FIRAT16, GC13.replace("[1]", "[2]")],
  "problem": "Firat&#39;s framework with the HDy transform: f(x) = bin(P(.|x)) builds a per-class "
      "posterior PMF. X is the matrix of class histograms, y the test histogram.",
  "algorithm": {"text": "Solve the constrained regression with a Hellinger objective on the simplex:",
      "math": ["minimize_p   1 - sum_i sqrt( y_i * (X p)_i )",
               "subject to   p >= 0 ,  sum_k p_k = 1"],
      "after": "Equivalent to per-bin matching aggregated through the simplex solve; reduces to HDy "
               "in the binary case."},
  "assumptions": "Posterior histograms must be shift-invariant per class and distinct across classes. "
      "Preferred over running HDy One-vs-All for genuine multiclass problems.",
  "relationship": "Multiclass HDy. Shares the constrained-regression backbone with GACC/GPACC, "
      "differing in transform (histogram) and loss (Hellinger).",
 },
 # ---- GHDx
 {"name": "GHDx", "title": "Generalized Hellinger Distance x",
  "imp": "from mlquantify.matching import GHDx",
  "summary": "Generalized Hellinger Distance x (GHDx) quantifier.",
  "description": "Targets prior probability shift. The multiclass, classifier-free form of HDx: it "
      "builds per-feature histograms and solves on the simplex for the prevalence matching the test "
      "feature histograms under a Hellinger objective. Needs only raw features.",
  "params": [("bins", "histogram bin count(s) per feature."),
             ("solver", "Simplex optimiser, default <font face='Courier'>'slsqp'</font>.")],
  "attrs": [("representation_", "Per-class, per-feature histograms."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Classifier-free; concatenates per-feature histogram blocks and matches them jointly on "
           "the simplex.",
  "see_also": "HDx, GHDy",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import GHDx",
              "&gt;&gt;&gt; q = GHDx().fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.34, 1: 0.33, 2: 0.33}",
              "&gt;&gt;&gt; q.aggregate(X_test)",
              "{0: 0.34, 1: 0.33, 2: 0.33}"],
  "refs": [FIRAT16, GC13.replace("[1]", "[2]")],
  "problem": "Firat&#39;s framework with the HDx transform f(x) = per-feature histogram, matched "
      "without any classifier.",
  "algorithm": {"text": "Concatenate per-feature histogram blocks and minimise the Hellinger "
      "objective on the simplex:",
      "math": ["minimize_p   1 - sum_i sqrt( y_i * (X p)_i )",
               "subject to   p >= 0 ,  sum_k p_k = 1",
               "(X stacks the per-feature class histogram blocks)"],
      "after": "No cross-validation; the per-feature blocks are built directly from the training "
               "features."},
  "assumptions": "Each feature&#39;s class distribution must be shift-invariant and informative. "
      "Use when no trustworthy classifier is available; weak with many noisy features.",
  "relationship": "Multiclass, feature-space counterpart of GHDy; the classifier-free branch of the "
      "unified framework.",
 },
 # ---- GKDEyML
 {"name": "GKDEyML", "title": "Generalized KDEy &mdash; Maximum Likelihood",
  "imp": "from mlquantify.matching import GKDEyML",
  "summary": "Generalized KDEy Maximum Likelihood (GKDEyML) quantifier.",
  "description": "Targets prior probability shift. A compose-based maximum-likelihood KDE quantifier: "
      "per-class kernel densities feed a mixture-likelihood objective solved on the simplex. The "
      "generalised, pluggable-representation sibling of KDEyML.",
  "params": [("estimator", "A probabilistic classifier (posteriors)."),
             ("bandwidth", "float, KDE bandwidth."),
             ("solver", "Simplex optimiser used to maximise the likelihood.")],
  "attrs": [("estimator_", "The fitted underlying classifier."),
            ("representation_", "Per-class fitted KDE models."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Shares KDEyML&#39;s objective but is wired through the compose / likelihood machinery, so "
           "the representation and loss can be swapped.",
  "see_also": "KDEyML, KDEyHD, KDEyCS",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import GKDEyML",
              "&gt;&gt;&gt; q = GKDEyML(LogisticRegression(), bandwidth=0.1).fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.25, 1: 0.40, 2: 0.35}",
              "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
              "&gt;&gt;&gt; q.aggregate(proba)",
              "{0: 0.25, 1: 0.40, 2: 0.35}"],
  "refs": [MOREO24],
  "problem": "Same simplex / posterior-density setting as KDEyML, expressed through the "
      "likelihood-compose interface.",
  "algorithm": {"text": "Maximise the mixture log-likelihood of the test posteriors under the "
      "per-class KDEs:",
      "math": ["maximize_p   sum_{x in test}  log( sum_i p_i * KDE_i(x) )",
               "subject to   p >= 0 ,  sum_i p_i = 1"],
      "after": "Identical objective to KDEyML; the compose form lets the density representation be "
               "replaced."},
  "assumptions": "As KDEyML: shift-invariant posteriors and a sensible bandwidth; benefits from "
      "calibrated probabilities. For multiclass likelihood-based quantification.",
  "relationship": "The compose-framework rendering of KDEyML, paired with the likelihood losses used "
      "by EMQ and MLPE.",
 },
 # ---- EDy
 {"name": "EDy", "title": "Energy Distance y",
  "imp": "from mlquantify.matching import EDy",
  "summary": "Energy Distance y (EDy) quantifier.",
  "description": "Targets prior probability shift. A distribution-matching quantifier that represents "
      "each class by the full set of its classifier predictions and minimises the energy distance "
      "between the test set and the prevalence-weighted mixture of class sets. Native multiclass and "
      "strong in the multiclass regime.",
  "params": [("estimator", "A probabilistic classifier (predictions)."),
             ("metric", ("Ground distance delta between predictions used in the energy term; "
                 "default Manhattan.", [
                 ("manhattan", "L1 distance between prediction vectors; the paper's default for EDy."),
                 ("euclidean", "L2 distance between prediction vectors."),
                 ("cityblock", "alias of Manhattan as accepted by scipy cdist.")])),
             ("solver", "Simplex optimiser, default <font face='Courier'>'slsqp'</font>.")],
  "attrs": [("estimator_", "The fitted underlying classifier."),
            ("representation_", "Per-class distance representation."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "The energy objective is a quadratic form p^T(2q&minus;Mp); distributions are estimated "
           "via cross-validation for both train and test. EDy uses classifier predictions, EDx uses "
           "raw features.",
  "see_also": "EDx, MMD_RKHS",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import EDy",
              "&gt;&gt;&gt; q = EDy(LogisticRegression()).fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.2, 1: 0.5, 2: 0.3}",
              "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
              "&gt;&gt;&gt; q.aggregate(proba)",
              "{0: 0.2, 1: 0.5, 2: 0.3}"],
  "refs": [DELCOZ22, KAWA16],
  "problem": "Prior shift matched with the energy distance. Model the test set T as the mixture "
      "D&#39; = sum_l p_l * D_cl of the per-class prediction sets; minimise the energy distance "
      "ED(T, D&#39;) over the prevalence p, with delta a ground distance between predictions.",
  "algorithm": {"text": "Dropping the test-only term, the energy distance becomes a quadratic form "
      "in the prevalence:",
      "math": ["minimize_p   2 * sum_l p_l * E[ delta(x_l, x_test) ]",
               "                - sum_l sum_l' p_l p_l' * E[ delta(x_l, x_l') ]",
               "   ==  p^T ( 2 q - M p )       (mlquantify EnergyLoss)",
               "   q_l  = mean delta(class l, test)",
               "   M_ll'= mean delta(class l, class l')",
               "subject to   p >= 0 ,  sum_l p_l = 1"],
      "after": "EDy sets delta on classifier predictions delta(h(x_i), h(x_j)); EDx sets it on raw "
               "features. Solved on the simplex."},
  "assumptions": "Assumes prediction (or feature) distributions are shift-invariant per class. "
      "Particularly effective for multiclass tasks; cost grows with the number of cross "
      "distances. Calibrated posteriors help EDy.",
  "relationship": "Energy-distance sibling of MMD_RKHS (kernel norm vs energy distance). EDx is the "
      "feature-space version using the original Kawakubo energy-distance estimator.",
 },
 # ---- EDx
 {"name": "EDx", "title": "Energy Distance x",
  "imp": "from mlquantify.matching import EDx",
  "summary": "Energy Distance x (EDx) quantifier.",
  "description": "Targets prior probability shift. The classifier-free energy-distance quantifier: it "
      "computes the energy distance directly on raw feature vectors between the test set and the "
      "prevalence mixture of per-class sets. Multiclass; needs no scorer.",
  "params": [("metric", "Ground distance on features, default Euclidean/Manhattan."),
             ("solver", "Simplex optimiser, default <font face='Courier'>'slsqp'</font>.")],
  "attrs": [("representation_", "Per-class distance representation in feature space."),
            ("classes_", "Class labels seen during fit.")],
  "notes": "Energy distance on raw features (Kawakubo et al.); avoids classifier calibration but "
           "ignores any learned representation.",
  "see_also": "EDy, MMD_RKHS",
  "example": ["&gt;&gt;&gt; from mlquantify.matching import EDx",
              "&gt;&gt;&gt; q = EDx().fit(X_train, y_train)",
              "&gt;&gt;&gt; q.predict(X_test)",
              "{0: 0.34, 1: 0.33, 2: 0.33}",
              "&gt;&gt;&gt; q.aggregate(X_test)",
              "{0: 0.34, 1: 0.33, 2: 0.33}"],
  "refs": [KAWA16.replace("[2]", "[1]"), DELCOZ22.replace("[1]", "[2]")],
  "problem": "Same energy-distance matching as EDy, but the ground distance delta acts on raw "
      "feature vectors, so no classifier is needed.",
  "algorithm": {"text": "Minimise the same energy quadratic form with feature-space distances:",
      "math": ["minimize_p   p^T ( 2 q - M p )",
               "   q_l  = mean delta_features(class l, test)",
               "   M_ll'= mean delta_features(class l, class l')",
               "subject to   p >= 0 ,  sum_l p_l = 1"],
      "after": "Distances are computed in input space; estimator-free."},
  "assumptions": "Feature distributions must be shift-invariant per class and meaningfully metric. "
      "Use when a classifier is unavailable or untrusted; sensitive to feature scaling.",
  "relationship": "Feature-space version of EDy and the original energy-distance estimator that EDy "
      "adapts to classifier predictions.",
 },
])
print("matching complete:", len(matching["entries"]), "entries")

# ============================================================================
SAER02 = "[1] Saerens, M., Latinne, P., &amp; Decaestecker, C. (2002). Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. <i>Neural Computation</i>, 14(1), 21&ndash;41."
ALEX20 = "[2] Alexandari, A., Kundaje, A., &amp; Shrikumar, A. (2020). Maximum Likelihood with Bias-Corrected Calibration is Hard-to-Beat at Label Shift Adaptation. <i>ICML</i>."
XUE09  = "[1] Xue, J. C., &amp; Weiss, G. M. (2009). Quantification and Semi-Supervised Classification Methods for Handling Changes in Class Distribution. <i>KDD</i>."
BARR13 = "[1] Barranquero, J., Gonz&aacute;lez, P., D&iacute;ez, J., &amp; del Coz, J. J. (2013). On the Study of Nearest Neighbor Algorithms for Prevalence Estimation in Binary Problems. <i>Pattern Recognition</i>, 46(2), 472&ndash;482."
PG17   = "[1] P&eacute;rez-G&aacute;llego, P., Quevedo, J. R., &amp; del Coz, J. J. (2017). Using Ensembles for Problems with Characterizable Changes in Data Distribution. <i>Information Fusion</i>, 34, 87&ndash;100."
PG19   = "[2] P&eacute;rez-G&aacute;llego, P., Casta&ntilde;o, A., Quevedo, J. R., &amp; del Coz, J. J. (2019). Dynamic Ensemble Selection for Quantification Tasks. <i>Information Fusion</i>, 45, 1&ndash;15."
MS25   = "[1] Moreo, A., &amp; Salvati, M. (2025). An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification. <i>LQ 2025 / arXiv</i>."

likelihood = {
 "name": "Likelihood", "short": "Likelihood",
 "intro": "Likelihood methods estimate prevalence by fitting the test priors that best explain the "
          "observed test data &mdash; either by maximising a mixture likelihood or by iteratively "
          "re-estimating the class distribution. They adjust a fixed classifier&#39;s outputs "
          "without retraining it.",
 "entries": [
  # ---- EMQ
  {"name": "EMQ", "title": "Expectation-Maximisation Quantifier",
   "imp": "from mlquantify.likelihood import EMQ",
   "summary": "Expectation-Maximisation Quantifier (EMQ) quantifier.",
   "description": "Targets prior probability shift. Runs the Saerens-Latinne-Decaestecker EM "
       "procedure: it alternately rescales the classifier posteriors by the current prior estimate "
       "and re-averages them, converging to the test priors that maximise the test-set likelihood. "
       "Requires soft, well-calibrated posteriors; no classifier retraining.",
   "params": [("estimator", "A probabilistic classifier with <font face='Courier'>predict_proba</font>."),
              ("max_iter", "int, maximum EM iterations."),
              ("tol", "float, convergence threshold on the prevalence change between iterations.")],
   "attrs": [("estimator_", "The fitted underlying classifier."),
             ("priors_", "Training class prevalences used to initialise EM."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "EMQ (a.k.a. SLD) is highly sensitive to calibration: bias-corrected temperature scaling "
            "of the posteriors before EM (Alexandari 2020) makes it hard to beat. It corrects "
            "classifier bias without retraining.",
   "see_also": "CDE, MLPE",
   "example": ["&gt;&gt;&gt; from mlquantify.likelihood import EMQ",
               "&gt;&gt;&gt; q = EMQ(LogisticRegression()).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.31, 1: 0.69}",
               "&gt;&gt;&gt; proba = q.estimator_.predict_proba(X_test)",
               "&gt;&gt;&gt; q.aggregate(proba)",
               "{0: 0.31, 1: 0.69}"],
   "refs": [SAER02, ALEX20],
   "problem": "Prior shift where only the priors p(y) change. Given a classifier trained with priors "
       "p_tr(y) and its posteriors P(y|x), EM seeks new priors p_te(y) that maximise the likelihood "
       "of the unlabelled test bag under the corrected posteriors.",
   "algorithm": {"text": "Initialise the priors to the training priors and iterate the E and M "
       "steps until convergence:",
       "steps": ["<b>E-step</b> &mdash; rescale posteriors by the current prior ratio and renormalise.",
                 "<b>M-step</b> &mdash; set the new prior of each class to the mean adjusted posterior.",
                 "Repeat until the prior estimate stops changing (tol) or max_iter is reached."],
       "math": ["E:  P^(s)(y|x) =  [ (p^(s)(y)/p_tr(y)) * P(y|x) ]",
                "               /  sum_k (p^(s)(k)/p_tr(k)) * P(k|x)",
                "M:  p^(s+1)(y) =  (1/n) * sum_{x in test} P^(s)(y|x)"],
       "after": "Converges to a local maximum of the test log-likelihood, usually within a few "
                "iterations; the final priors are the prevalence estimate."},
   "assumptions": "Unbiased under prior shift provided the posteriors are calibrated on the test "
       "distribution; mis-calibration is its main failure mode. One of the strongest quantifiers "
       "when paired with calibration.",
   "relationship": "Likelihood counterpart of the distribution-matching family. KDEyML maximises a "
       "similar mixture likelihood with KDE-smoothed densities; MLPE is the no-iteration baseline.",
  },
  # ---- CDE
  {"name": "CDE", "title": "CDE-Iterate",
   "imp": "from mlquantify.likelihood import CDE",
   "summary": "Class Distribution Estimation by iteration (CDE) quantifier.",
   "description": "Targets prior probability shift. Iteratively re-estimates the test class "
       "distribution and rebuilds the classifier with a cost ratio reflecting the current estimate, "
       "shifting the effective decision threshold toward the test distribution at each pass. "
       "Binary base method.",
   "params": [("estimator", "A cost-sensitive base classifier."),
              ("max_iter", "int, number of CDE iterations."),
              ("tol", "float, convergence threshold on the estimate.")],
   "attrs": [("estimator_", "The classifier from the last iteration."),
             ("priors_", "Training class prevalences."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "Each iteration adjusts the FP/FN cost ratio from the estimated Pos:Neg ratio, so the "
            "decision boundary tracks the test distribution. Usually converges in a few iterations.",
   "see_also": "EMQ, MLPE",
   "example": ["&gt;&gt;&gt; from mlquantify.likelihood import CDE",
               "&gt;&gt;&gt; q = CDE(LogisticRegression()).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.40, 1: 0.60}",
               "&gt;&gt;&gt; preds = q.estimator_.predict(X_test)",
               "&gt;&gt;&gt; q.aggregate(preds, y_train=y_train)",
               "{0: 0.40, 1: 0.60}"],
   "refs": [XUE09],
   "problem": "Binary prior shift where the changed class distribution biases a fixed classifier. "
       "CDE estimates the new distribution and feeds it back as a cost ratio to re-bias the model.",
   "algorithm": {"text": "Iteratively classify, estimate, and re-weight:",
       "steps": ["Build the initial classifier on the original labelled data and classify the test set.",
                 "Estimate the new class distribution (Pos:Neg) from those predictions.",
                 "Rebuild / re-threshold the classifier with a cost ratio derived from the estimate.",
                 "Repeat for a few iterations; return the final distribution."],
       "math": ["cost_ratio <- Pos2Neg(original) adjusted by current NEW estimate",
                "C_i        <- build_classifier( data , cost_ratio )",
                "p_hat       <- distribution of C_i on the test set"],
       "after": "Each pass moves the effective threshold toward the test distribution, refining the "
                "prevalence."},
   "assumptions": "Assumes prior shift and a cost-sensitive learner. Can oscillate if the classifier "
       "is unstable; best when the base learner responds smoothly to cost weighting. Binary.",
   "relationship": "Like EMQ it iteratively reconciles predictions with an evolving prior, but it "
       "re-biases the classifier (threshold/costs) rather than rescaling posteriors analytically.",
  },
  # ---- MLPE
  {"name": "MLPE", "title": "Maximum Likelihood Prevalence Estimation",
   "imp": "from mlquantify.likelihood import MLPE",
   "summary": "Maximum Likelihood Prevalence Estimation (MLPE) quantifier.",
   "description": "Assumes no shift. Predicts the training class prevalence for every test bag &mdash; "
       "the maximum-likelihood estimate when the test prior is assumed equal to the training prior. "
       "Ignores the test features entirely; serves as the trivial lower-bound baseline.",
   "params": [("estimator", "Optional; unused for prediction (kept for a uniform interface).")],
   "attrs": [("priors_", "Training class prevalences returned as the estimate."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "MLPE is the &#39;no-change&#39; reference: any useful quantifier should beat it under "
            "real shift. Its error equals the shift magnitude.",
   "see_also": "EMQ, CDE",
   "example": ["&gt;&gt;&gt; from mlquantify.likelihood import MLPE",
               "&gt;&gt;&gt; q = MLPE().fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)        # always the training prevalence",
               "{0: 0.50, 1: 0.50}",
               "&gt;&gt;&gt; q.aggregate(y_train=y_train)",
               "{0: 0.50, 1: 0.50}"],
   "refs": ["[1] Gonz&aacute;lez, P., Casta&ntilde;o, A., Chawla, N. V., &amp; del Coz, J. J. (2017). A Review on Quantification Learning. <i>ACM Computing Surveys</i>, 50(5), 74."],
   "problem": "Degenerate case of prior estimation: assume p_te(y) = p_tr(y). The multinomial "
       "likelihood of the (unobserved) test labels is then maximised by the training frequencies.",
   "algorithm": {"text": "Return the training prevalence regardless of the test bag:",
       "math": ["p_hat(y) = p_tr(y) = (1/n_tr) * sum_{x in train} I( y_x = y )"],
       "after": "No optimisation and no use of the test data."},
   "assumptions": "Correct only when there is no shift. Useful purely as a baseline and sanity "
       "check; never use it when the test distribution is expected to move.",
   "relationship": "The null model of the likelihood family &mdash; EMQ and CDE reduce to it when the "
       "test bag carries no evidence of shift.",
  },
 ],
}

neighbors = {
 "name": "Neighbors", "short": "Neighbors",
 "intro": "Neighbour methods estimate prevalence from a nearest-neighbour vote, reweighting the "
          "votes to undo the majority-class bias that ordinary k-NN inherits under class imbalance.",
 "entries": [
  {"name": "PWK", "title": "Proportion-Weighted k-NN",
   "imp": "from mlquantify.neighbors import PWK",
   "summary": "Proportion-Weighted k-Nearest-Neighbour (PWK) quantifier.",
   "description": "Targets prior probability shift. Classifies test instances with a k-NN whose "
       "neighbour votes are reweighted to compensate the bias toward the majority class, then counts "
       "the weighted predictions. The weighting (PWKa, exponent a) interpolates between plain k-NN "
       "and full prevalence-compensation. Binary base method.",
   "params": [("n_neighbors", "int, number of neighbours k."),
              ("alpha", "float, weighting exponent a (a=1 = PWK; large a -&gt; plain k-NN)."),
              ("metric", "Distance metric for the neighbour search.")],
   "attrs": [("classifier_", "The fitted weighted k-NN classifier."),
             ("classes_", "Class labels seen during fit.")],
   "notes": "Weights are inversely related to each class&#39;s training prevalence, so the minority "
            "class is not swamped; PWK is among the few NN quantifiers to beat CC/AC significantly.",
   "see_also": "CC, ACC",
   "example": ["&gt;&gt;&gt; from mlquantify.neighbors import PWK",
               "&gt;&gt;&gt; q = PWK(n_neighbors=10).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.45, 1: 0.55}",
               "&gt;&gt;&gt; preds = q.classifier_.predict(X_test)",
               "&gt;&gt;&gt; q.aggregate(preds, y_train=y_train)",
               "{0: 0.45, 1: 0.55}"],
   "refs": [BARR13],
   "problem": "Binary prior shift. Ordinary k-NN counts are biased toward the majority class because "
       "dense regions dominate the neighbour sets. PWK rebalances the vote before counting.",
   "algorithm": {"text": "Weight each neighbour&#39;s vote to offset its class&#39;s training "
       "prevalence, classify, then classify-and-count:",
       "math": ["w(class c) ~ ( prior_tr(c) ) ^ ( -1/a )    (down-weight majority class)",
                "label(x) = argmax_c  sum_{j in kNN(x)} w(class of j) * I(class j = c)",
                "p_hat(c) = (1/n) * sum_{x in test} I( label(x) = c )"],
       "after": "a=1 recovers PWK; as a grows the weights flatten to 1 and PWK -&gt; standard k-NN."},
   "assumptions": "Assumes prior shift and a meaningful distance metric. Robust and simple; like all "
       "k-NN it suffers in high dimensions and needs feature scaling. Binary focus.",
   "relationship": "A counting method whose base predictor is a bias-corrected k-NN rather than a "
       "thresholded scorer; PWKa generalises it with KNN and PWK as the extremes of a.",
  },
 ],
}
print("likelihood + neighbors defined")

# ============================================================================
QUADAPT25 = "[1] Ortega, J. P., Luth Junior, L. F., Zalewski, W., &amp; Maletzke, A. (2025). QuaDapt: Drift-Resilient Quantification via Parameters Adaptation. <i>Proc. 5th Int. Workshop on Learning to Quantify (LQ 2025)</i>, p. 64."
MOSS21    = "[2] Maletzke, A., dos Reis, D., Hassan, W., &amp; Batista, G. (2021). Accurately Quantifying under Score Variability. <i>ICDM 2021</i>, 1228&ndash;1233. (Model for Score Simulation, MoSS.)"

meta = {
 "name": "Meta", "short": "Meta",
 "intro": "Meta quantifiers wrap a base quantifier to add a higher-level capability: ensemble "
          "diversity and selection, bootstrap-based uncertainty, or on-the-fly adaptation to "
          "distribution shift. They compose with any of the methods above.",
 "entries": [
  # ---- EnsembleQ
  {"name": "EnsembleQ", "title": "Ensemble Quantifier",
   "imp": "from mlquantify.meta import EnsembleQ",
   "summary": "Ensemble Quantifier with prevalence-controlled diversity (EnsembleQ).",
   "description": "Targets prior probability shift (and characterisable distribution change). Trains "
       "many copies of a base quantifier on subsamples drawn at deliberately different class "
       "prevalences, then aggregates their estimates &mdash; optionally selecting only the members "
       "whose training distribution resembles the test sample (dynamic selection). Reduces variance "
       "and adapts to the test shift.",
   "params": [("quantifier", "The base quantifier replicated across ensemble members."),
              ("size", "int, default=50. Number of ensemble members."),
              ("min_prop, max_prop", "Prevalence range used when sampling training batches; sets the diversity."),
              ("selection_metric", ("Which members vote at prediction time.", [
                  ("all", "use every member (plain bagged average; no selection)."),
                  ("ptr", "keep members whose training prevalence is closest to an initial test estimate."),
                  ("ds", "keep members whose training score distribution is closest to the test distribution.")])),
              ("p_metric", "Fraction of members retained when a selection metric (ptr/ds) is used; "
                  "smaller keeps only the most relevant members."),
              ("protocol", ("Protocol used to draw each member's training prevalence.", [
                  ("uniform", "prevalences spread uniformly over the simplex (default)."),
                  ("artificial", "fixed grid of prevalences (artificial-prevalence protocol)."),
                  ("natural", "prevalences resampled from the natural training distribution."),
                  ("kraemer", "Kraemer sampling of prevalence vectors on the simplex.")])),
              ("return_type", ("Aggregation over the selected members.", [
                  ("mean", "average of the member estimates (default)."),
                  ("median", "median of the member estimates; more robust to outlier members.")]))],
   "attrs": [("models", "Fitted ensemble member quantifiers."),
             ("train_prevalences", "Training prevalence of each member."),
             ("classes", "Class labels seen during fit.")],
   "notes": "Members are trained with sampling-with-replacement so that p(x|y) is preserved while "
            "only p(y) varies. Dynamic selection (&#39;ptr&#39;/&#39;ds&#39;) is what lets the "
            "ensemble specialise to the test shift; with &#39;all&#39; it is a plain bagged average.",
   "see_also": "AggregativeBootstrap, QuaDapt",
   "example": ["&gt;&gt;&gt; from mlquantify.meta import EnsembleQ",
               "&gt;&gt;&gt; from mlquantify.matching import DyS",
               "&gt;&gt;&gt; q = EnsembleQ(DyS(LogisticRegression()), size=50).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.49, 1: 0.51}"],
   "refs": [PG17, PG19],
   "problem": "Prior shift whose magnitude is unknown at training time. A single quantifier is tuned "
       "to one training prevalence; an ensemble spanning many prevalences brackets the possible test "
       "distributions and lets the relevant members dominate.",
   "algorithm": {"text": "Build a prevalence-diverse ensemble, then aggregate (optionally select):",
       "steps": ["For m members, draw a training sample at a target prevalence (sampling with "
                 "replacement, preserving p(x|y)); train a base quantifier on each.",
                 "At test time, optionally select the members whose training distribution is most "
                 "similar to the test sample (ptr: prevalence similarity; ds: score-distribution "
                 "similarity).",
                 "Aggregate the selected members&#39; prevalence estimates by mean or median."],
       "math": ["members:  q_1, ..., q_m  trained at prevalences  pi_1, ..., pi_m",
                "select:    S = top-p_metric members by similarity( pi_j , test )",
                "p_hat = aggregate_{j in S}  q_j(test)"],
       "after": "Dynamic selection adapts the ensemble to each test bag without retraining."},
   "assumptions": "Assumes the test shift is &#39;characterisable&#39; &mdash; covered by the range "
       "of training prevalences sampled. Most useful when the shift is large or unknown; cost scales "
       "with the ensemble size. Pair with a strong base quantifier (e.g. DyS, EMQ).",
   "relationship": "A wrapper, not a new estimator: it amplifies any base method (counting, matching "
       "or likelihood) with diversity and dynamic selection.",
  },
  # ---- AggregativeBootstrap
  {"name": "AggregativeBootstrap", "title": "Aggregative Bootstrap",
   "imp": "from mlquantify.meta import AggregativeBootstrap",
   "summary": "Aggregative Bootstrap quantifier for prevalence confidence regions.",
   "description": "Targets prior probability shift. Wraps an aggregative quantifier and bootstrap-"
       "resamples its predictions to turn a point prevalence estimate into a confidence region. "
       "Because aggregative quantifiers classify once and aggregate, the resampling is applied to "
       "the cheap aggregation step, giving uncertainty estimates efficiently.",
   "params": [("quantifier", "The base aggregative quantifier to wrap."),
              ("n_train_bootstraps", "int, resamples of the training predictions."),
              ("n_test_bootstraps", "int, resamples of the test predictions."),
              ("region_type", ("Shape of the confidence region built from the resampled estimates.", [
                  ("intervals", "independent per-class percentile intervals (default; simplest to read)."),
                  ("ellipse", "Gaussian confidence ellipse in the simplex; captures class correlations."),
                  ("ellipse-clr", "ellipse in centered-log-ratio (Aitchison) space; respects simplex geometry.")])),
              ("confidence_level", "float, default=0.95. Probability mass the region is meant to cover.")],
   "attrs": [("train_predictions", "Cached predictions on the training/validation set."),
             ("y_train", "Labels for the cached predictions."),
             ("classes", "Class labels seen during fit.")],
   "notes": "Combining train- and test-side resampling captures both model and sampling uncertainty. "
            "The CLR ellipse lives in Aitchison (log-ratio) space, respecting the simplex geometry.",
   "see_also": "EnsembleQ, ConfidenceInterval",
   "example": ["&gt;&gt;&gt; from mlquantify.meta import AggregativeBootstrap",
               "&gt;&gt;&gt; from mlquantify.likelihood import EMQ",
               "&gt;&gt;&gt; q = AggregativeBootstrap(EMQ(LogisticRegression()),",
               "...                          n_train_bootstraps=10, n_test_bootstraps=10).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)        # point estimate (+ stored region)",
               "{0: 0.31, 1: 0.69}"],
   "refs": [MS25],
   "problem": "A point prevalence estimate gives no sense of its reliability. Bootstrap resampling "
       "approximates the sampling distribution of the estimate, from which a confidence region at a "
       "chosen level is read off.",
   "algorithm": {"text": "Exploit the two-phase structure of aggregative quantifiers:",
       "steps": ["Run the expensive classification phase once, caching the train and test predictions.",
                 "Resample the cached predictions with replacement (train and/or test sides) and "
                 "re-run only the aggregation, producing many prevalence estimates.",
                 "Summarise the resampled estimates as a point estimate plus a confidence region "
                 "(per-class interval, simplex ellipse, or CLR-space ellipse)."],
       "math": ["for b = 1..B:  p_hat^(b) = aggregate( resample(predictions) )",
                "region = level-set of { p_hat^(1), ..., p_hat^(B) } at confidence_level"],
       "after": "Applying bootstrap only to aggregation gives large speed-ups versus resampling the "
                "whole pipeline."},
   "assumptions": "Inherits the base quantifier&#39;s prior-shift assumption; the region is valid in "
       "so far as the bootstrap approximates the true sampling distribution (needs enough test "
       "points and resamples). Use whenever a decision needs an uncertainty band, not just a number.",
   "relationship": "A wrapper that adds uncertainty quantification to any aggregative method; it "
       "produces the confidence regions defined in the confidence module.",
  },
  # ---- QuaDapt
  {"name": "QuaDapt", "title": "Drift-Resilient Quantification via Parameters Adaptation",
   "imp": "from mlquantify.meta import QuaDapt",
   "summary": "QuaDapt: adaptive quantification via synthetic score simulation.",
   "description": "Targets general distribution shift / concept drift, not only prior shift. Wraps a "
       "soft base quantifier and, at prediction time, simulates training score distributions with "
       "MoSS at several overlap levels, selecting the one whose mixed scores best match the test "
       "scores; that synthetic set becomes the aggregation reference. Binary base method "
       "(One-vs-Rest for multiclass).",
   "params": [("quantifier", "A soft (probabilistic) base aggregative quantifier."),
              ("measure", ("Distance comparing each synthetic score set with the test scores, used "
                  "to pick the best merging factor; default <font face='Courier'>'topsoe'</font>.", [
                  ("topsoe", "symmetric information-theoretic distance (default)."),
                  ("hellinger", "bounded sqrt-probability distance."),
                  ("probsymm", "probabilistic symmetric chi-square distance."),
                  ("sord", "Sample Ordinal Distance on the raw scores (bin-free).")])),
              ("merging_factors", "Candidate MoSS overlap levels m to evaluate; default "
                  "0.1&ndash;0.9 in steps of 0.2. Each m sets how much the synthetic positive and "
                  "negative scores overlap (0 = well separated, 1 = fully overlapping)."),
              ("strategy", ("Multiclass decomposition into binary sub-problems.", [
                  ("ovr", "one-vs-rest: one binary adaptation per class (default)."),
                  ("ovo", "one-vs-one: one binary adaptation per class pair.")]))],
   "attrs": [("classes_", "Class labels seen during fit."),
             ("y_train", "Training labels stored at fit.")],
   "notes": "Built on MoSS (Model for Score Simulation): the merging factor m sets class-score "
            "overlap (0 = well separated, 1 = heavy overlap). Adapting m to the test scores makes a "
            "standard quantifier resilient when the score-distribution complexity drifts.",
   "see_also": "DyS, EnsembleQ",
   "example": ["&gt;&gt;&gt; from mlquantify.meta import QuaDapt",
               "&gt;&gt;&gt; from mlquantify.matching import DyS",
               "&gt;&gt;&gt; q = QuaDapt(DyS(LogisticRegression())).fit(X_train, y_train)",
               "&gt;&gt;&gt; q.predict(X_test)",
               "{0: 0.49, 1: 0.51}",
               "&gt;&gt;&gt; proba = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)",
               "&gt;&gt;&gt; q.aggregate(proba, y_train)",
               "{0: 0.49, 1: 0.51}"],
   "refs": [QUADAPT25, MOSS21],
   "problem": "Beyond prior shift, the very complexity (overlap) of the class score distributions "
       "can drift between train and test, breaking quantifiers that assume fixed class-conditional "
       "scores. QuaDapt makes no assumption on the drift form; it realigns the training reference to "
       "the observed test scores.",
   "algorithm": {"text": "MoSS generates synthetic positive/negative scores parameterised by a "
       "merging factor m (overlap) and a positive proportion alpha:",
       "math": ["MoSS(n, alpha, m) = syn(+, floor(alpha*n), m)  U  syn(-, floor((1-alpha)*n), m)",
                "syn(+, n, m) = { X_i ^ m } ,        X_i ~ U(0,1)",
                "syn(-, n, m) = { 1 - X_i ^ m } ,    X_i ~ U(0,1)"],
       "steps": ["For each candidate merging factor m, build MoSS synthetic positive/negative scores.",
                 "Pick the m whose mixed synthetic distribution is closest to the test scores under "
                 "the chosen distance (topsoe by default).",
                 "Use that synthetic, drift-aligned score set as the training reference for the base "
                 "quantifier&#39;s aggregate, and estimate the prevalence."],
       "after": "Only the quantifier&#39;s internal score reference is adapted &mdash; no classifier "
                "retraining. Generalises DySyn (DyS + MoSS) to any classifier-based quantifier."},
   "assumptions": "Does not assume a specific drift type; assumes scores can be realigned by a "
       "one-parameter overlap model (MoSS). Most valuable under concept/covariate drift that changes "
       "score complexity, where prior-shift quantifiers fail. Binary core, OvR for multiclass.",
   "relationship": "A drift-resilient wrapper for any soft quantifier; extends Maletzke et al.&#39;s "
       "MoSS-based DySyn from DyS to the whole quantifier family.",
  },
 ],
}
print("meta defined:", len(meta["entries"]), "entries")

# ============================================================================
# UTILITY RENDERER (solvers / representations / losses)
# ============================================================================
def render_util(e):
    fl = []
    head = [anchor(f"{e['name']} — {e['title']}", 2),
            para(f"{e['name']} &mdash; {e['title']}", "Entry"),
            para(f"{e['kind']} &nbsp;|&nbsp; <font face='Courier'>{e['imp']}</font>", "EntryTag")]
    head.append(surface_banner("API&nbsp;&nbsp;&middot;&nbsp;&nbsp;interface", SLATE))
    head.append(para("Summary", "Sub"))
    head.append(para(e["summary"], "Body"))
    fl.append(KeepTogether(head))
    fl.append(para("Description", "Sub"))
    fl.append(para(e["description"], "Body"))
    fl.append(para("Signature &amp; parameters", "Sub"))
    fl.append(kv_table(e["params"]))
    fl.append(para("Returns" if e.get("returns") else "Attributes", "Sub"))
    fl.append(kv_table(e.get("returns") or e.get("attrs", [])))
    fl.append(para("Example", "Sub"))
    fl.append(code_block(e["example"]))
    # User guide
    fl.append(Spacer(1, 2))
    fl.append(surface_banner("USER GUIDE&nbsp;&nbsp;&middot;&nbsp;&nbsp;role &amp; theory", ACCENT2))
    fl.append(para("Role &amp; mechanism", "Sub"))
    fl.append(para(e["role"], "Body"))
    if e.get("math"):
        fl.append(math_block(e["math"]))
    fl.append(para("Used by", "Sub"))
    fl.append(para(e["used_by"], "Body"))
    if e.get("refs"):
        fl.append(para("References", "Sub"))
        for r in e["refs"]:
            fl.append(para(r, "Ref"))
    fl.append(rule(5, 7))
    return fl

def render_util_family(fam):
    fl = [anchor(fam["name"], 1), para(fam["name"], "Family"), rule(2, 4),
          para(fam["intro"], "FamilyIntro")]
    for e in fam["entries"]:
        e["kind"] = fam["kind"]
        fl += render_util(e)
    return fl

solvers = {
 "name": "Solvers", "kind": "solver",
 "intro": "Solvers are the optimisation backends shared by the quantifier families. They turn a "
          "distribution-matching or likelihood objective into a prevalence vector, either over the "
          "binary interval [0,1] or over the full probability simplex.",
 "entries": [
  {"name": "solve_binary", "title": "Binary prevalence solver",
   "imp": "from mlquantify.solvers import solve_binary",
   "summary": "Minimise a scalar objective over the binary prevalence space [0, 1].",
   "description": "Optimises a one-dimensional objective f(alpha), where alpha is the positive-class "
       "prevalence, by exhaustive grid search, ternary search (unimodal objectives), or scipy&#39;s "
       "bounded scalar minimiser. Returns the two-class prevalence vector and the objective value.",
   "params": [("objective", "callable f(alpha) -&gt; float, alpha in [0,1]."),
              ("solver", ("Optimisation strategy over alpha.", [
                  ("auto", "selects <font face='Courier'>'bounded'</font> (default)."),
                  ("grid", "evaluate the objective at grid_size evenly-spaced points; robust, no unimodality needed."),
                  ("ternary", "interval-trisection search; fastest when the objective is unimodal."),
                  ("bounded", "scipy bounded scalar minimiser.")])),
              ("grid_size", "int, number of candidate alpha values for the grid solver."),
              ("tol", "float, convergence tolerance for the ternary solver.")],
   "returns": [("prevalence", "ndarray (2,) = [1 - alpha, alpha]."),
               ("loss", "float, objective value at the optimum.")],
   "example": ["&gt;&gt;&gt; from mlquantify.solvers import solve_binary",
               "&gt;&gt;&gt; p, loss = solve_binary(lambda a: (a - 0.3)**2, solver='bounded')",
               "&gt;&gt;&gt; round(p[1], 2)",
               "0.3"],
   "role": "The backbone of the binary matching methods. The dissimilarity between the alpha-mixed "
       "class representation and the test representation is unimodal in alpha for most distances, so "
       "ternary or bounded search converges quickly and robustly.",
   "math": ["alpha* = argmin_{0 <= alpha <= 1}  objective(alpha)",
            "return [ 1 - alpha* , alpha* ]"],
   "used_by": "DyS, HDy, HDx, SORD, SMM and any binary distribution-matching quantifier.",
   "refs": [DYS19],
  },
  {"name": "ternary_search", "title": "Ternary search",
   "imp": "from mlquantify.solvers import ternary_search",
   "summary": "Find the minimum of a unimodal function by interval trisection.",
   "description": "Repeatedly evaluates the objective at two interior probes and discards the third "
       "of the interval that cannot contain the minimum, until the interval is narrower than tol.",
   "params": [("left, right", "float, bounds of the search interval."),
              ("objective", "callable, unimodal on [left, right]."),
              ("tol", "float, stop when right - left &le; tol.")],
   "returns": [("minimum", "float, approximate location of the minimum.")],
   "example": ["&gt;&gt;&gt; from mlquantify.solvers import ternary_search",
               "&gt;&gt;&gt; round(ternary_search(0.0, 1.0, lambda x: (x-0.3)**2), 4)",
               "0.3"],
   "role": "A derivative-free line search that exploits unimodality: each iteration shrinks the "
       "interval by one third, giving linear convergence without gradients.",
   "math": ["m1 = left + (right-left)/3 ;  m2 = right - (right-left)/3",
            "if f(m1) < f(m2): right = m2  else: left = m1   # repeat until width <= tol"],
   "used_by": "solve_binary (the &#39;ternary&#39; strategy) and, through it, the DyS family.",
  },
  {"name": "solve_simplex", "title": "Simplex (SLSQP) solver",
   "imp": "from mlquantify.solvers import solve_simplex",
   "summary": "Minimise a function over the probability simplex using SLSQP.",
   "description": "Minimises a multivariate objective f(p) subject to the simplex constraints "
       "(p &ge; 0, sum p = 1) via sequential least-squares programming; the result is clipped to "
       "non-negative values and renormalised.",
   "params": [("objective", "callable f(p) -&gt; float, p of shape (n_classes,)."),
              ("n_classes", "int, dimensionality of p."),
              ("x0", "optional initial guess (defaults to uniform)."),
              ("random_state", "seed for a random simplex start.")],
   "returns": [("prevalence", "ndarray (n_classes,) summing to 1."),
               ("loss", "float, objective value at the optimum.")],
   "example": ["&gt;&gt;&gt; from mlquantify.solvers import solve_simplex",
               "&gt;&gt;&gt; target = np.array([0.2, 0.5, 0.3])",
               "&gt;&gt;&gt; p, _ = solve_simplex(lambda p: np.sum((p-target)**2), n_classes=3)",
               "&gt;&gt;&gt; np.round(p, 2)",
               "array([0.2, 0.5, 0.3])"],
   "role": "The multiclass workhorse: it enforces the constrained-regression feasibility set of the "
       "unified framework, returning a valid distribution for any smooth objective.",
   "math": ["minimize_p  objective(p)",
            "subject to  sum_k p_k = 1 ,  0 <= p_k <= 1   (SLSQP)"],
   "used_by": "GACC, GPACC, FM, GHDy, GHDx, GKDEyML, MMD_RKHS, EDy, EDx, KDEy (multiclass).",
   "refs": [FIRAT16],
  },
  {"name": "minimize_prevalence", "title": "Prevalence minimisation dispatcher",
   "imp": "from mlquantify.solvers import minimize_prevalence",
   "summary": "Minimise an objective over the simplex, dispatching binary vs multiclass.",
   "description": "Routes to the binary solver when n_classes == 2 and the requested strategy is "
       "compatible, and to the SLSQP simplex solver otherwise. With solver='auto' it picks bounded "
       "search for binary problems and SLSQP for multiclass.",
   "params": [("objective", "callable; scalar for binary, vector for multiclass."),
              ("n_classes", "int &ge; 2."),
              ("solver", ("Backend selection; binary strategies dispatch to solve_binary, "
                  "<font face='Courier'>'slsqp'</font> to solve_simplex.", [
                  ("auto", "bounded for binary, SLSQP for multiclass (default)."),
                  ("grid", "binary grid search (n_classes = 2 only)."),
                  ("ternary", "binary ternary search (n_classes = 2 only)."),
                  ("bounded", "binary bounded scalar minimiser (n_classes = 2 only)."),
                  ("slsqp", "constrained SLSQP over the full simplex (any n_classes).")])),
              ("grid_size, tol, x0, random_state", "forwarded to the chosen backend.")],
   "returns": [("prevalence", "ndarray (n_classes,) summing to 1."),
               ("loss", "float, objective value at the optimum.")],
   "example": ["&gt;&gt;&gt; from mlquantify.solvers import minimize_prevalence",
               "&gt;&gt;&gt; p, _ = minimize_prevalence(lambda a: (a-0.3)**2, n_classes=2, solver='bounded')",
               "&gt;&gt;&gt; round(p[1], 2)",
               "0.3"],
   "role": "A single entry point so a quantifier can be written once and run in both binary and "
       "multiclass mode; it hides the binary/simplex distinction behind one call.",
   "math": ["n_classes == 2 and solver in {grid,ternary,bounded}  ->  solve_binary",
            "otherwise (slsqp)                                    ->  solve_simplex"],
   "used_by": "Every optimisation-based quantifier that must support both binary and multiclass.",
  },
  {"name": "minimize_prevalence_blocks", "title": "Block-wise prevalence minimisation",
   "imp": "from mlquantify.solvers import minimize_prevalence_blocks",
   "summary": "Minimise a loss per representation block, then aggregate the estimates.",
   "description": "For each sub-vector (block) of the test and training representations &mdash; e.g. "
       "each histogram bin group &mdash; builds a block-specific objective, minimises it "
       "independently, and aggregates the per-block prevalence estimates by median or mean.",
   "params": [("objective_factory", "callable producing a block objective from (test_block, train_block)."),
              ("test/train_representations", "the full representations to split into blocks."),
              ("block_slices", "slice objects identifying each block."),
              ("aggregate", "'median' (default) or 'mean'.")],
   "returns": [("prevalence", "ndarray (n_classes,), the aggregated estimate.")],
   "example": ["&gt;&gt;&gt; from mlquantify.solvers import minimize_prevalence_blocks",
               "&gt;&gt;&gt; # one estimate per histogram-bin block, aggregated by the median",
               "&gt;&gt;&gt; p = minimize_prevalence_blocks(factory, q_test, q_train,",
               "...                                  block_slices, n_classes=2)"],
   "role": "Implements the &#39;sweep and take the median&#39; recipe of the histogram-matching "
       "family: each bin configuration yields one estimate and the median is robust to the "
       "unreliable ones.",
   "math": ["for each block b:  p_b = argmin  objective_b( test_b , train_b )",
            "p_hat = median_b ( p_b )      (or mean)"],
   "used_by": "HDy, HDx and the histogram distribution-matching quantifiers.",
   "refs": [GC13],
  },
 ],
}
print("solvers defined")

representations = {
 "name": "Representations", "kind": "representation",
 "intro": "Representations turn a sample of instances (or scores) into the fixed-length descriptor "
          "that a distribution-matching quantifier compares. Swapping the representation is what "
          "turns one matching skeleton into HDy, EDy, MMD or KDEy.",
 "entries": [
  {"name": "HistogramRepresentation", "title": "Histogram representation",
   "imp": "from mlquantify.representations import HistogramRepresentation",
   "summary": "Per-feature histogram representation with optional block partitioning.",
   "description": "Bins each feature (or score) independently and concatenates the normalised bin "
       "frequencies into one vector. Supports fixed or quantile edges, one-hot mode, Laplace "
       "smoothing, and block slices for per-feature recovery.",
   "params": [("bins", "int or array of ints &mdash; how many bins each feature/score is split into. "
                  "More bins capture finer distribution detail but need more samples to fill "
                  "reliably; passing several values lets a quantifier sweep resolutions and take a "
                  "median. This is the resolution knob of the representation."),
              ("range", "(low, high) &mdash; the value interval the bins span, default (0.0, 1.0) "
                  "since classifier posteriors live in [0,1]. Values outside the range fall into the "
                  "edge bins. Set it to the actual feature range when binning raw features."),
              ("mode", ("How each value is turned into bin mass.", [
                  ("histogram", "count values per bin, then normalise to a probability mass function (default)."),
                  ("onehot", "assign each value to its bin as a one-hot vector and average; a softer per-sample encoding.")])),
              ("features", "indices of the features to histogram; <font face='Courier'>None</font> "
                  "uses all. Lets you restrict matching to informative features (e.g. only the "
                  "positive-class posterior in the binary case)."),
              ("bin_edges", ("How the bin boundaries are placed &mdash; the key accuracy/robustness "
                  "trade-off of the histogram.", [
                  ("fixed", "equal-width edges from <font face='Courier'>range</font>; fast, no fitting, but wastes bins where data is sparse (default)."),
                  ("auto", "data-driven (quantile) edges learned at fit time so each bin holds a similar mass; more robust to skewed scores at the cost of storing the edges.")])),
              ("partition_blocks", "if <font face='Courier'>True</font>, keep each feature's bins as a "
                  "separate contiguous block and expose <font face='Courier'>block_slices_</font>, so "
                  "a quantifier can match (and aggregate) one feature-block at a time instead of one "
                  "big concatenated vector. This is what enables the per-bin median sweep of HDy/HDx."),
              ("laplace_smoothing", "if <font face='Courier'>True</font>, add a small (1/k) count to "
                  "every bin before normalising. This removes zero bins, which otherwise make "
                  "ratio/log-based distances (Hellinger, Topsoe) unstable when a bin is empty in one "
                  "distribution but not the other.")],
   "attrs": [("class_representations_", "Per-class histogram vectors."),
             ("block_slices_", "Per-feature slices (when partition_blocks=True).")],
   "example": ["&gt;&gt;&gt; from mlquantify.representations import HistogramRepresentation",
               "&gt;&gt;&gt; rep = HistogramRepresentation(bins=(8,)).fit(scores, y)",
               "&gt;&gt;&gt; rep.transform(scores[:10]).shape",
               "(8,)"],
   "role": "Discretises a distribution into a probability-mass vector that can be linearly mixed by "
       "a candidate prevalence and compared with a histogram distance &mdash; the classic mixture-"
       "model representation.",
   "math": ["H(S)_i = (# values of S in bin i) / |S|     (normalised PMF)",
            "mix(alpha) = alpha*H(S+) + (1-alpha)*H(S-)"],
   "used_by": "DyS, HDy, HDx, GHDy, GHDx.",
   "refs": [GC13],
  },
  {"name": "KDERepresentation", "title": "Kernel density representation",
   "imp": "from mlquantify.representations import KDERepresentation",
   "summary": "Per-class kernel density estimate over instances/posteriors.",
   "description": "Fits a kernel density estimator per class (bandwidth, kernel) and exposes per-class "
       "likelihoods of test points; the test-time representation is the raw vector. A smooth, "
       "multivariate alternative to histograms.",
   "params": [("bandwidth", "float &mdash; the smoothing radius of the density estimate and the "
                  "single most important parameter. Too small and each class density spikes on its "
                  "training points (over-fitting); too large and the class densities blur together, "
                  "flattening the differences the quantifier relies on."),
              ("kernel", ("shape of the local bump placed on each training point; sets how mass "
                  "decays with distance. Default <font face='Courier'>'gaussian'</font>.", [
                  ("gaussian", "smooth, infinite-support bell; the usual default."),
                  ("tophat", "flat box: equal weight inside the bandwidth, zero outside."),
                  ("epanechnikov", "parabolic, finite support; theoretically efficient."),
                  ("exponential", "exponential decay with no hard cutoff."),
                  ("linear", "triangular weight decreasing linearly to the bandwidth edge."),
                  ("cosine", "cosine-shaped weight over the bandwidth.")]))],
   "attrs": [("class_representations_", "Per-class fitted KernelDensity models.")],
   "example": ["&gt;&gt;&gt; from mlquantify.representations import KDERepresentation",
               "&gt;&gt;&gt; rep = KDERepresentation(bandwidth=0.2).fit(X, y)",
               "&gt;&gt;&gt; rep.class_representations_[0]",
               "KernelDensity(bandwidth=0.2)"],
   "role": "Models each class density continuously, avoiding the bin-sparsity problem of histograms "
       "on the multiclass posterior simplex.",
   "math": ["KDE_i(x) = (1/n_i) sum_{x_j in class i} K_h( x - x_j )",
            "mixture(x) = sum_i p_i * KDE_i(x)"],
   "used_by": "KDEyML, KDEyHD, KDEyCS, GKDEyML.",
   "refs": [MOREO24],
  },
  {"name": "DistanceRepresentation", "title": "Distance representation",
   "imp": "from mlquantify.representations import DistanceRepresentation",
   "summary": "Mean pairwise-distance descriptor of a sample to each class.",
   "description": "Summarises a set of instances by its mean pairwise distances to each training "
       "class, yielding an (n_classes,) descriptor &mdash; the basis of the energy-distance objective.",
   "params": [("metric", ("the ground distance between instances, passed to scipy "
                  "<font face='Courier'>cdist</font>; it defines what &#39;close&#39; means for the "
                  "energy objective. Default <font face='Courier'>'euclidean'</font>.", [
                  ("euclidean", "straight-line L2 distance (default)."),
                  ("cityblock", "Manhattan / L1 distance; the choice used by EDy on predictions."),
                  ("cosine", "1 minus cosine similarity; scale-invariant, good for sparse vectors.")]))],
   "attrs": [("class_representations_", "Mean class-to-training distances."),
             ("X_train_, y_train_", "Stored training data.")],
   "example": ["&gt;&gt;&gt; from mlquantify.representations import DistanceRepresentation",
               "&gt;&gt;&gt; rep = DistanceRepresentation(metric='cityblock').fit(X, y)",
               "&gt;&gt;&gt; rep.transform(X_test).shape",
               "(n_classes,)"],
   "role": "Provides the cross-distance terms q and M that make up the energy-distance quadratic "
       "form, without binning or density estimation.",
   "math": ["q_l  = mean delta(class l, test)",
            "M_ll'= mean delta(class l, class l')"],
   "used_by": "EDy (on predictions), EDx (on features).",
   "refs": [DELCOZ22],
  },
  {"name": "KernelMeanRepresentation", "title": "Kernel mean embedding",
   "imp": "from mlquantify.representations import KernelMeanRepresentation",
   "summary": "Mean embedding of a sample in a reproducing-kernel Hilbert space.",
   "description": "Represents a set of instances by its kernel mean embedding (the column-wise mean "
       "feature map), exact under a linear kernel and approximate under non-linear kernels.",
   "params": [("kernel", ("the kernel whose feature map the sample is embedded under; determines "
                  "which moments of the distribution are captured. Default <font face='Courier'>'rbf'</font>.", [
                  ("rbf", "Gaussian kernel; universal, captures all moments (default)."),
                  ("linear", "embedding is just the mean vector; captures only the first moment."),
                  ("poly", "polynomial kernel; captures moments up to its degree."),
                  ("sigmoid", "hyperbolic-tangent kernel.")])),
              ("gamma", "scale of the rbf/poly/sigmoid kernel; <font face='Courier'>None</font> "
                  "defaults to 1/n_features. Controls how quickly similarity decays with distance."),
              ("degree", "polynomial degree (poly kernel only) &mdash; the highest moment matched."),
              ("coef0", "independent term in the poly/sigmoid kernel; shifts the similarity floor.")],
   "attrs": [("class_representations_", "Per-class mean embeddings.")],
   "example": ["&gt;&gt;&gt; from mlquantify.representations import KernelMeanRepresentation",
               "&gt;&gt;&gt; rep = KernelMeanRepresentation(kernel='rbf').fit(X, y)",
               "&gt;&gt;&gt; emb = rep.transform(X_test)"],
   "role": "Embeds a whole distribution as a single point in an RKHS, so distribution matching "
       "reduces to matching mean embeddings (the MMD objective).",
   "math": ["Phi_bar(S) = (1/|S|) sum_{x in S} Phi(x)     (mean embedding)",
            "match: sum_y theta_y Phi_bar_y  ~  Phi_bar_test"],
   "used_by": "MMD_RKHS.",
   "refs": [IYER14],
  },
  {"name": "PredictionRepresentation", "title": "Prediction representation",
   "imp": "from mlquantify.representations import PredictionRepresentation",
   "summary": "Posterior / label representation (soft mean or hard class-frequency).",
   "description": "Maps posteriors or labels to either the class-mean posterior ('soft') or the "
       "one-hot class-frequency vector ('hard', equivalent to Classify-and-Count on the "
       "representation). Custom transforms and a nested representation are supported. "
       "HardPredictionRepresentation and SoftPredictionRepresentation are the fixed-mode variants.",
   "params": [("method", ("how classifier outputs become the descriptor (ignored when "
                  "<font face='Courier'>func</font> is given).", [
                  ("soft", "average the posterior vectors &mdash; the class-mean posterior (used by GPACC)."),
                  ("hard", "one-hot the argmax label and average &mdash; the class-frequency vector, i.e. Classify-and-Count (used by GACC).")])),
              ("average", "if <font face='Courier'>True</font> return the descriptor averaged over "
                  "instances; if <font face='Courier'>False</font> return the per-instance matrix "
                  "(needed when a downstream representation does its own pooling)."),
              ("func", "optional callable <font face='Courier'>func(X, representation) -&gt; Z</font> "
                  "replacing the built-in transform; lets you plug a custom mapping."),
              ("representation", "optional nested representation applied to the transformed output "
                  "before returning &mdash; e.g. feed soft posteriors into a histogram.")],
   "attrs": [("priors_", "Empirical class proportions seen during fit.")],
   "example": ["&gt;&gt;&gt; from mlquantify.representations import PredictionRepresentation",
               "&gt;&gt;&gt; rep = PredictionRepresentation(method='soft').fit(proba, y)",
               "&gt;&gt;&gt; rep.transform(proba_test)"],
   "role": "Turns classifier outputs into the mean-posterior or class-frequency vectors that the "
       "constrained-regression methods match on the simplex.",
   "math": ["soft: z = (1/n) sum_x P(.|x)            (mean posterior)",
            "hard: z = (1/n) sum_x onehot(argmax P(.|x))   (= CC)"],
   "used_by": "GACC (hard), GPACC (soft), FM, and compose-based matching.",
   "refs": [FIRAT16],
  },
 ],
}

losses = {
 "name": "Losses", "kind": "loss",
 "intro": "Losses are the objective functions minimised by the solvers. A quantifier family is "
          "largely defined by the (representation, loss, solver) triple it composes; the same loss "
          "is reused across methods.",
 "entries": [
  {"name": "DistanceLoss", "title": "Distribution distance loss",
   "imp": "from mlquantify.losses import DistanceLoss",
   "summary": "Symmetric distance between two probability distributions.",
   "description": "Computes a histogram/distribution dissimilarity (hellinger, topsoe, prob_symm, "
       "sqEuclidean, euclidean) between a mixture and a target, optionally normalising both to valid "
       "distributions first.",
   "params": [("distance", ("the dissimilarity minimised between the two distributions; default "
                  "<font face='Courier'>'hellinger'</font>.", [
                  ("hellinger", "bounded distance over sqrt-probabilities; the classic HDy choice."),
                  ("topsoe", "symmetric information-theoretic distance; best general performer in DyS."),
                  ("probsymm", "probabilistic symmetric chi-square distance."),
                  ("sqEuclidean", "squared Euclidean distance between the vectors."),
                  ("euclidean", "Euclidean (L2) distance &mdash; the square root of sqEuclidean.")])),
              ("normalize", "if <font face='Courier'>True</font>, clip and renormalise both inputs to "
                  "valid distributions first, so the distance is well defined even on raw counts.")],
   "returns": [("loss", "float, the distance value.")],
   "example": ["&gt;&gt;&gt; from mlquantify.losses import get_loss",
               "&gt;&gt;&gt; loss = get_loss('topsoe')",
               "&gt;&gt;&gt; round(loss([0.3, 0.7], [0.5, 0.5]), 4)",
               "0.0528"],
   "role": "The objective minimised by histogram mixture models: find the prevalence whose mixed "
       "histogram is closest to the test histogram under this distance.",
   "math": ["hellinger(P,Q) = sqrt( 1 - sum_i sqrt(P_i Q_i) )",
            "topsoe(P,Q)    = sum_i P_i ln(2P_i/(P_i+Q_i)) + Q_i ln(2Q_i/(P_i+Q_i))"],
   "used_by": "DyS, HDy, HDx (and any selectable-distance matcher).",
   "refs": [DYS19],
  },
  {"name": "LeastSquaresLoss", "title": "Least-squares loss",
   "imp": "from mlquantify.losses import LeastSquaresLoss",
   "summary": "Squared Euclidean (L2) loss for constrained regression.",
   "description": "Computes ||target - M&middot;mixture||^2; with no mixing matrix M it reduces to "
       "||target - mixture||^2. The objective of the unified constrained-regression framework.",
   "params": [("mixture", "estimated prevalence vector."),
              ("target", "target representation vector."),
              ("M", "optional mixing matrix applied as M @ mixture.")],
   "returns": [("loss", "float, the squared norm.")],
   "example": ["&gt;&gt;&gt; from mlquantify.losses import get_loss",
               "&gt;&gt;&gt; loss = get_loss('least_squares')",
               "&gt;&gt;&gt; loss([0.4, 0.6], [0.5, 0.5])",
               "0.02"],
   "role": "Implements y = X p in least-squares form: minimising it on the simplex inverts the "
       "per-class transform matrix to recover the prevalence.",
   "math": ["L(p) = || y - X p ||^2",
            "minimised over  p >= 0 ,  sum p = 1"],
   "used_by": "GACC, GPACC, FM, MMD_RKHS.",
   "refs": [FIRAT16],
  },
  {"name": "HellingerSurrogateLoss", "title": "Hellinger surrogate loss",
   "imp": "from mlquantify.losses import HellingerSurrogateLoss",
   "summary": "Gradient-friendly surrogate for the squared Hellinger distance.",
   "description": "Returns -sum_i sqrt(p_i q_i); minimising it is equivalent to minimising the "
       "squared Hellinger distance but is numerically better behaved for gradient-free solvers.",
   "params": [("mixture, target", "the two vectors to compare."),
              ("M", "optional mixing matrix."),
              ("normalize", "normalise before computing.")],
   "returns": [("loss", "float; lower (more negative) is better.")],
   "example": ["&gt;&gt;&gt; from mlquantify.losses import get_loss",
               "&gt;&gt;&gt; loss = get_loss('hellinger_surrogate')",
               "&gt;&gt;&gt; round(loss([0.3, 0.7], [0.5, 0.5]), 4)",
               "-0.9747"],
   "role": "Lets the multiclass Hellinger matchers be optimised on the simplex with SLSQP by "
       "dropping the constant and square-root-of-sum that make the raw Hellinger awkward.",
   "math": ["H^2(p,q) = 1 - sum_i sqrt(p_i q_i)",
            "argmin H^2  ==  argmin ( - sum_i sqrt(p_i q_i) )"],
   "used_by": "GHDy, GHDx, KDEyHD.",
   "refs": [GC13],
  },
  {"name": "EnergyLoss", "title": "Energy-distance loss",
   "imp": "from mlquantify.losses import EnergyLoss",
   "summary": "Quadratic energy-distance objective for distribution matching.",
   "description": "Computes the quadratic form p^T(2q - M p), where q is the cross-distance vector "
       "(class-to-test) and M the pairwise class energy-distance matrix. Minimising it minimises the "
       "energy distance between the mixture and the test distribution.",
   "params": [("prevalence", "estimated prevalence p."),
              ("q", "mean class-to-test distances."),
              ("M", "pairwise class-to-class distance matrix.")],
   "returns": [("loss", "float, the energy-distance value.")],
   "example": ["&gt;&gt;&gt; from mlquantify.losses import get_loss",
               "&gt;&gt;&gt; loss = get_loss('energy')",
               "&gt;&gt;&gt; loss([0.5, 0.5], [0.4, 0.6], [[0,1],[1,0]])",
               "0.5"],
   "role": "The closed quadratic that the energy-distance quantifiers minimise on the simplex, with "
       "q and M precomputed from the distance representation.",
   "math": ["L(p) = p^T ( 2 q - M p )",
            "q_l = mean delta(class l, test) ;  M_ll' = mean delta(class l, class l')"],
   "used_by": "EDy, EDx.",
   "refs": [DELCOZ22],
  },
  {"name": "NegativeLogLikelihoodLoss", "title": "Negative log-likelihood loss",
   "imp": "from mlquantify.losses import NegativeLogLikelihoodLoss",
   "summary": "Negative log-likelihood of mixture likelihoods (mean or sum).",
   "description": "Computes -log p(x) elementwise and reduces by mean or sum. The companion "
       "MixtureNegativeLogLikelihoodLoss and RegularizedMixtureNLLLoss build the mixture likelihood "
       "from per-class likelihoods and add optional simplex-smoothness penalties.",
   "params": [("likelihood", "per-sample mixture likelihoods in (0, 1]."),
              ("reduction", ("how the per-sample negative log-likelihoods are combined into one "
                  "scalar.", [
                  ("mean", "average over samples; comparable across test bags of different sizes (default)."),
                  ("sum", "total over samples; matches the raw joint log-likelihood.")]))],
   "returns": [("loss", "float, the reduced negative log-likelihood.")],
   "example": ["&gt;&gt;&gt; from mlquantify.losses import get_loss",
               "&gt;&gt;&gt; loss = get_loss('nll')",
               "&gt;&gt;&gt; round(loss(np.array([0.5, 0.5])), 4)",
               "0.6931"],
   "role": "The likelihood objective behind EM and KDE-ML quantifiers: choose the prevalence that "
       "maximises the likelihood of the test bag under the mixture density.",
   "math": ["L(p) = - sum_{x in test} log( sum_i p_i * lik_i(x) )",
            "argmin L  ==  maximum-likelihood prevalence"],
   "used_by": "EMQ, KDEyML, GKDEyML, MLPE.",
   "refs": [SAER02],
  },
 ],
}
print("representations + losses defined")

# ============================================================================
# PAGE DECORATION + BUILD
# ============================================================================
PAGE_W, PAGE_H = A4
MARGIN = 2.5 * cm

def _cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#fafcfd"))
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H - 0.55 * cm, PAGE_W, 0.55 * cm, fill=1, stroke=0)
    canvas.rect(0, 0, PAGE_W, 0.55 * cm, fill=1, stroke=0)
    canvas.restoreState()

def _body_page(canvas, doc):
    canvas.saveState()
    # header
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 1.35 * cm, "mlquantify — Full Documentation")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.35 * cm, "API Docstrings & User Guide")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, PAGE_H - 1.5 * cm, PAGE_W - MARGIN, PAGE_H - 1.5 * cm)
    # footer
    canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.0 * cm, "v0.3.1")
    canvas.drawCentredString(PAGE_W / 2, 1.0 * cm,
                             "mlquantify Reference Guide — Methods · Solvers · Representations · Losses")
    canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, "%d" % doc.page)
    canvas.restoreState()

def build():
    doc = BaseDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.0 * cm, bottomMargin=1.8 * cm,
        title="mlquantify — Full Documentation (Methods, Solvers, Representations & Losses)",
        author="Luiz Fernando Luth Junior",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=_cover_page),
        PageTemplate(id="body", frames=[frame], onPage=_body_page),
    ])

    story = []
    story += [NextPageTemplate("cover")] + cover()
    story += [NextPageTemplate("body")]
    story += how_to_read()

    # Part I: Methods
    story += [PageBreak()] + part_divider("I", "Quantification Methods",
        "The 33 quantifiers, organised by family. Each is documented as an API Docstring "
        "(interface, no math) followed by a User Guide (problem, objective, assumptions, "
        "relationships). The shift assumption and primary paper appear on both surfaces.")
    for fam in (counting, matching, likelihood, neighbors, meta):
        story.append(PageBreak())
        story += render_family(fam)

    # Part II: Solvers
    story += [PageBreak()] + part_divider("II", "Solvers",
        "Optimisation backends that convert a matching or likelihood objective into a "
        "prevalence vector over [0,1] or the probability simplex.")
    story.append(PageBreak())
    story += render_util_family(solvers)

    # Part III: Representations
    story += [PageBreak()] + part_divider("III", "Representations",
        "Feature/score descriptors that a distribution-matching quantifier compares; the "
        "representation choice is what specialises a matching skeleton into HDy, EDy, MMD or KDEy.")
    story.append(PageBreak())
    story += render_util_family(representations)

    # Part IV: Losses
    story += [PageBreak()] + part_divider("IV", "Losses",
        "Objective functions minimised by the solvers. A quantifier family is largely the "
        "(representation, loss, solver) triple it composes.")
    story.append(PageBreak())
    story += render_util_family(losses)

    doc.build(story)
    print("WROTE", OUT)

if __name__ == "__main__":
    build()
