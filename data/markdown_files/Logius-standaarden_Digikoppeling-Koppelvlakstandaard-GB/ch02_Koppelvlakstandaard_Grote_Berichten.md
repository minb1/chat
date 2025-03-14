# Koppelvlakstandaard Grote Berichten

## Inleiding

De situatie kan zich voordoen dat een Digikoppeling bericht een grootte krijgt
die niet meer efficiënt door de Digikoppeling adapters verwerkt kan worden. Ook kan
het zich voordoen dat er behoefte bestaat aan het buiten de normale procesgang
('out-of-band') uitwisselen van grote hoeveelheden informatie tussen systemen.
In die gevallen zal dit “grote bericht” op een andere wijze verstuurd moeten
worden: middels de Digikoppeling Koppelvlakstandaard Grote Berichten. De
volgende aanpak wordt dan gehanteerd:

- De verzender stelt een bestand samen uit (een deel van) de gegevens die
    normaliter in het “grote bericht” verzonden zou worden. Het resultaat wordt
    aangeduid met de term “groot bestand”. Merk op dat dit ook een “groot” xml
    bestand kan zijn, een CAD bestand, een PDF document, een ZIP bestand, et
    cetera.

- De verzender stelt metadata samen over het grote bestand en deelt deze
    metadata in een Digikoppeling-bericht [in een zgn. stuurbericht].

- Uitwisseling van het grote bestand vindt plaats via een PULL of een PUSH
    principe.   
    Bij Het PULL principe biedt de verzender het groot bestand aan via een Grote
    Berichten File service aan de ontvanger.  
    Bij het PUSH principe stuurt de verzender het groot bestand naar de Grote
    Berichten File service van de ontvanger

- De bestandsoverdracht is niet “betrouwbaar”; hiervoor dient de ontvanger
    aanvullende maatregelen te implementeren (retry-mechanisme,
    foutafhandeling). De Koppelvlakstandaard bevat hiervoor handvatten.
    Toepassing van deze handvatten in concrete implementaties vallen buiten de
    scope van het koppelvlak.

Merk op dat het stuurbericht naast metadata ook voorzien kan zijn van
inhoudelijke informatie die al nodig is bij de verwerking van het bericht.

Dit document beschrijft welke gegevens er in de metadata opgenomen moeten worden
en hoe het HTTP 1.1 protocol gebruikt moet worden voor de overdracht van het
grote bestand.

## Nieuw in deze versie

In deze versie wordt de Digikoppeling Grote Berichten PUSH variant
geïntroduceerd, naast de reeds bestaande PULL variant. We hebben ervoor gekozen
de beschrijving van de PULL variant te integreren in de bestaande PUSH versie,
omdat de voorwaarden en regels voor beide richtingen vrijwel identiek zijn.

## Verzenden van Grote Berichten

### Pull Principe

Het principe is dat de verzender het grote bestand aanbiedt via een Grote
Berichten File Service en een bericht stuurt aan de ontvanger dat het bericht
geplaatst is, de ontvanger kan het bestand vervolgens ophalen.

![Uitwisseling groot bestand via Grote Berichten file service van de zender](media/UitwisselingGrootbestandviaGBfileservice_zender.png "Uitwisseling groot bestand via Grote Berichten file service van de zender")

In bovenstaand figuur is dit grafisch weergegeven.

- Stap 1: De verzender verstuurt het bericht met de meta-data van het bestand,
bijvoorbeeld naam, locatie, grootte etc.

- Stap 2: De ontvanger ontvangt het bericht met de meta-data, en download en
verwerkt vervolgens het bestand (PULL).

Opmerking  
De verzender maakt hiervoor het te verzenden bestand gereed , eventueel wordt
dit in meerdere delen gesplitst als dit wenselijk is.

### Push Principe

Het principe is dat de verzender het grote bestand aanbiedt aan de Grote
Berichten File Service van de ontvanger (via een upload) en een bericht stuurt
aan de ontvanger dat het grote bestand verstuurd is, de ontvanger kan het
bestand vervolgens verwerken.

![Uitwisseling groot bestand via Grote Berichten file service van de ontvanger](media/UitwisselingGrootbestandviaGBfileservice_ontvanger.png "Uitwisseling groot bestand via Grote Berichten file service van de ontvanger")

In bovenstaand figuur is dit grafisch weergegeven.

- Stap 1: De verzender verstuurt het grote bericht naar de Grote berichten File
service van de ontvanger.

- Stap 2: De verzender verstuurt het bericht met de meta-data van het bestand,
bijvoorbeeld naam, locatie, grootte etc.

Opmerking  
De verzender maakt hiervoor het te verzenden bestand gereed, eventueel wordt dit
in meerdere delen gesplitst als dit wenselijk is.

## Gebruiksvoorwaarden

Voor het gebruik van het Digikoppeling Koppelvlakstandaard Grote Berichten
gelden onderstaande algemene eisen:

| **Referentie** | **Specificatie**                                                                                                                                                                                                                                                                                                                                                                                      |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VW000          | Partijen MOGEN bilateraal overeen komen bij welke MiB berichtomvang de standaard Grote Berichten van toepassing is of dat volstaan kan worden met een gewone Digikoppeling uitwisseling sec                                                                                                                                                                         |
|                | Een harde grens voor de berichtomvang is lastig te bepalen en in praktische zin is er sprake van overlap. Daarom is er voor gekozen dat partijen bilaterale afspraken kunnen maken waarin afgeweken wordt van de genoemde grens onder VW001, met dien verstande dat door het bilateraal karakter het nooit als argument gebruikt kan worden om andere organisaties te verplichten hieraan te voldoen. |
| VW001          | Als partijen niet tot overeenstemming komen MOETEN zij berichten groter dan 20 MiB via het Koppelvlak Grote Berichten afhandelen.                                                                                                                                                                                                                                                                     |
|                | Niet elke ontvanger is in staat om grote berichten te ontvangen (en te verwerken). Daarnaast dient te worden voorkomen dat grote berichten het transactionele berichtenverkeer eventueel zouden kunnen verstoren. Daarom dient ten aanzien van de omvang een harde grens te worden afgesproken.                                                                                                       |
| VW002          | Voor de overdracht van metadata MOET gebruik gemaakt worden van Digikoppeling, zoals aangeven in het hoofdstuk Metadata in dit document.                                                                                                                                                                                                                                                              |
| VW003          | Voor de overdracht van grote bestanden MOET gebruik gemaakt worden van het mechanisme zoals aangeven in het hoofdstuk Bestandsoverdracht in dit document.                                                                                                                                                                                                                                             |

