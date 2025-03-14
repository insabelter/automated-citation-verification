# citation-verification

Inspired by Tianmai M. Zhang, Neil F. Abernethy: "Detecting Reference Errors in Scientific Literature with Large Language Models" (https://arxiv.org/abs/2411.06101)

Ablauf der Implementierung:
- vorgegebene Daten analysieren (Diagramme erstellen)
- Versuch, PDFs automatisch zu downloaden mit Crawler - nicht erfolgreich
- Automatisch in Zotero importieren über Titel und DOI unter Verwendung von https://anystyle.io/
- Auto-Download der PDFs über Zotero, falls nicht gefunden manuell überprüfen
    - Für einige PDF kein Institutional Access (dann über Google, sci-hub gesucht)
    - Einige PDFs sind retracted 

Ideas for result improvements:
- use full GROBID model
- alayze and refactor extraction output
- use output_format variable for model prompts