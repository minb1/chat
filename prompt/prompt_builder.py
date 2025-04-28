# prompt/prompt_builder.py
import datetime

def create_cqr_prompt(user_query: str, chat_history: str) -> str:
    """
    Creates a prompt for an LLM to rewrite the user query into a standalone query,
    using the chat history for context. Focuses on context denoising.
    """
    # Basic history formatting, adjust if your redis_handler provides richer format
    history_str = chat_history if chat_history else "Dit is het eerste bericht in dit gesprek."

    prompt = f"""
    Gegeven de volgende chatgeschiedenis en de laatste gebruikersvraag, herschrijf de laatste vraag zodat deze volledig op zichzelf staat en geoptimaliseerd is voor het zoeken naar relevante documenten in een technische kennisbank. Verwijder eventuele ambiguïteit of afhankelijkheid van de vorige beurten. Geef **alleen** de herschreven, zelfstandige zoekvraag terug, zonder extra uitleg of opmaak.

    **Chat Geschiedenis:**
    {history_str}

    **Laatste Gebruikersvraag:**
    "{user_query}"

    **Herschreven Zoekvraag:**
    """
    return prompt


def create_prompt(user_query, context=None, chat_history=None, rewritten_query=None):
    """
    Creates the final prompt for answer generation, including context, history,
    and potentially the rewritten query for reference (though context is primary).
    """
    prompt = f"""
    Jij speelt de rol als behulpzame assistent om het Logius team te versterken met hun helpdesk.
    De vragen die je ontvangt zijn op basis van de documentatie die door Logius beheert word.
    Deze vragen zullen hoog-technisch zijn, en vereisen een antwoord dat diepgaande technisch inzicht vereist.
    Zorg ervoor dat je elke vraag beantwoord met een goede uitleg, en concrete voorbeelden, om ervoor te zorgen dat de
    vraag van de gebruiker goed beantwoord is. Om hiermee te helpen, wordt er context uit de documentatie gegeven.
    Zorg ervoor dat je tot zen zeerste alleen informatie uit die context gebruikt.
    Als het van toepassing is, is er ook een chat geschiedenis meegegeven, maar dat is optioneel.

    Beantwoord de **oorspronkelijke gebruikersvraag** op basis van de verstrekte context en chatgeschiedenis.
    Geef een zo volledig en nauwkeurig mogelijk antwoord.
    Let op technische details en vermeld uit welk document je het antwoord gehaald heb. De documentnaam komt als bestandspad binnen. Zet dit bestandspad ALTIJD om naar een menselijk-leesbaar formaat, zodat het makkelijk te herleiden is.
    Bijvoorbeeld "Sectie x uit het document y". De filepaths volgen het structuur van: "standaard/document/sectie". Vermeld de bestandspadden NIET in je response, alleen de leesbare vorm ervan. Noem de bestandsnaam of het pad zoals `standaard/x/y.md` **nooit** letterlijk in het antwoord.

    **Chat Geschiedenis (max 3 laatste beurten):**
    {chat_history if chat_history else "Geen relevante geschiedenis."}

    **Gevonden Context uit Documentatie:**
    --- CONTEXT START ---
    {context if context else "Geen relevante documenten gevonden."}
    --- CONTEXT EINDE ---

    **Oorspronkelijke Gebruikersvraag:**
    "{user_query}"
    """

    # Optional: Include rewritten query if you want the LLM to see it, but usually not necessary
    # if rewritten_query and rewritten_query != user_query:
    #     prompt += f"\n\n**(Ter informatie: De vraag is intern herschreven voor zoeken als: \"{rewritten_query}\")**"

    prompt += """
    Antwoord in het Nederlands. Geef alleen het antwoord op de vraag, begin niet met "Antwoord:" of een andere introductie. Zorg voor duidelijke referenties naar de bronnen zoals gevraagd.
    """
    return prompt


def HyDE(user_query):
    # ... (keep existing HyDE prompt) ...
    prompt = f"""
    Gegeven de volgende technische vraag of opdracht: "{user_query}"

    Voer deze opdracht uit als een HyDE implementatie.
    Schrijf een technische documentatie die de volgende onderdelen omvat:

    1. **Introductie & Overzicht**
       Geef een kort overzicht van het onderwerp en leg de context uit.

    2. **Systeemarchitectuur & Onderliggende Principes**
       Beschrijf de architectuur, de gebruikte technologieën en de kernprincipes.

    3. **Installatie- en Configuratie-instructies**
       Geef stap-voor-stap instructies voor de installatie en configuratie, inclusief vereisten.

    4. **Gebruiksscenario's & Voorbeelden**
       Presenteer concrete voorbeelden en scenario’s waarin de oplossing gebruikt kan worden.

    5. **Foutopsporing & Probleemoplossing**
       Bied richtlijnen voor het diagnosticeren en oplossen van veelvoorkomende problemen.

    Zorg ervoor dat de documentatie helder, technisch nauwkeurig en begrijpelijk is voor een technisch publiek.
    Geef alleen de gegenereerde documentatie terug, zonder extra uitleg of opmaak.
    """
    return prompt

def AugmentQuery(user_query):
    # ... (keep existing AugmentQuery prompt) ...
    prompt = f"""
    Herschrijf de volgende gebruikersvraag om deze effectiever te maken voor het zoeken in een technische documentatiedatabase.
    Denk aan synoniemen, gerelateerde technische termen, of het expliciteren van de onderliggende intentie.
    Focus op het verbeteren van de kans dat relevante documenten worden gevonden.
    Geef alleen de **éne** beste, herschreven zoekopdracht als resultaat, zonder extra uitleg of opmaak.

    Originele Vraag: "{user_query}"

    Herschreven Zoekopdracht:
    """
    return prompt
