from transformers import pipeline

def riassumi_transformers_astrattivo(testo):
    """Riassume un testo utilizzando un modello pre-allenato di Transformers (summarization)."""
    summarizer = pipeline("summarization") # Usa il modello predefinito per summarization
    riassunto = summarizer(testo, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return riassunto

testo_esempio = """
La Banca Centrale Europea (BCE) ha deciso di alzare i tassi di interesse di 0.50 punti percentuali.
Questa mossa è stata fatta per combattere l'inflazione elevata nella zona euro.
La presidente della BCE, Christine Lagarde, ha dichiarato che ulteriori aumenti dei tassi potrebbero essere necessari.
Gli analisti finanziari si aspettano che l'inflazione rimanga alta per il resto dell'anno.
L'aumento dei tassi di interesse potrebbe avere un impatto sui mutui e sui prestiti per le imprese.
"""

riassunto_transformers = riassumi_transformers_astrattivo(testo_esempio)
print(riassunto_transformers)