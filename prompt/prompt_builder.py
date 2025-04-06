

def create_prompt(user_query, context=None, chat_history=None):

    prompt = f"""
    Jij speelt de rol als behulpzame assistent om het Logius team te versterken met hun helpdesk. 
    De vragen die je ontvangt zijn op basis van de documentatie die door Logius beheert word.
    Deze vragen zullen hoog-technisch zijn, en vereisen een antwoord dat diepgaande technisch inzicht vereist.
    Zorg ervoor dat je elke vraag beantwoord met een goede uitleg, en concrete voorbeelden, om ervoor te zorgen dat de 
    vraag van de gebruiker goed beantwoord is. Om hiermee te helpen, wordt er context uit de documentatie gegeven.
    Zorg ervoor dat je tot zen zeerste alleen informatie uit die context gebruikt. 
    Als het van toepassing is, is er ook een chat geschiedenis meegegeven, maar dat is optioneel.

    Beantwoord de volgende vraag op basis van de verstrekte context.
    Geef een zo volledig en nauwkeurig mogelijk antwoord. 
    Let op technische details en vermeld uit welk document je het antwoord gehaald heb.

    **Vraag:** {user_query}

    **Context:** {context}

    **Vorige chatberichten:** {chat_history}
    Antwoord in het Nederlands.
    """
    return prompt


def HyDE(user_query):
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
    """
    return prompt
