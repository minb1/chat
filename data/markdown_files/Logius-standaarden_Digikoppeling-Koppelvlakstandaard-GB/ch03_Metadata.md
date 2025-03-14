# Metadata

De metadata beschrijft de informatie over het bestand dat verstuurd wordt met
HTTP 1.1. De metadata zelf wordt verzonden via een (WUS/ebMS/API) Digikoppeling Koppelvlak 

## Functionele beschrijving

De onderstaande regels zijn van toepassing.

| **Referentie** | **Specificatie**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MD000          | Metadata MOET verstuurd worden middels een Digikoppeling bericht.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| MD001          | De metadata XML-structuur MOET voldoen aan het XML schema in hoofdstuk Metadata XML Schema Definitie.  *De metadata kan een op zichzelf staand bericht zijn, maar ook een deel van een groter bericht. Het is daarbij ook toegestaan om meerdere grote bestanden in een bericht op te nemen; voor ieder afzonderlijk bestand dient dan afzonderlijk metadata in het bericht te worden opgenomen.*                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| MD002          | Voor ieder groot bestand MOET een unieke URL gegenereerd te worden; deze URL dient gebruikt te worden om het betreffende bestand op te halen. De URL is dus uniek voor het gehele Digikoppeling domein en wordt in het meta-bericht via het element `<location>` verstrekt aan de ontvanger. Bij een pull wordt hier `<senderUrl>` gevuld en bij push wordt hier `<receiverUrl>` gevuld.  *Door aan ieder bestand een unieke URL toe te kennen kan gegarandeerd worden dat het meta-bericht altijd aan het juiste bestand refereert. Het is wel toegestaan om hetzelfde bestand meerdere keren te verzenden (meerdere ontvangers); iedere ontvanger ontvangt dan wel een eigen meta-bericht, maar de URL verwijst dan telkens naar hetzelfde bestand. Ook is het toegestaan om meerdere unieke URL’s naar hetzelfde bestand te laten verwijzen.* |
| MD003          | De metadata MAG het moment aangeven (datum/tijd) waarop het grote bestand beschikbaar zal zijn (element `<creationTime>`) ALS dit veld ontbreekt of het moment ligt in het verleden MOET het bestand, uiterlijk op het moment dat de metadata verzonden wordt, beschikbaar zijn.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| MD004          | De metadata MAG het moment aangeven tot wanneer het grote bestand beschikbaar zal zijn. Het grote bestand MOET dan tenminste beschikbaar zijn tot het moment dat in de metadata aangegeven wordt (element `<expirationTime>`) na dat moment is de beschikbaarheid van het bestand niet meer gegarandeerd.  *Door een beperking op te leggen aan de beschikbaarheid wordt voorkomen dat het niet duidelijk is wanneer de betreffende bestanden weer mogen worden verwijderd.*                                                                                                                                                                                                                                                                                                                                                                   |
| MD005          | De metadata MOET aangeven hoe groot het bestand is, uitgedrukt in het aantal bytes (element `<size>`). Indien met deelbestanden wordt gewerkt: de metadata MOET aangeven hoeveel deelbestanden het zijn, hoe groot elk van de deelbestanden is zijn en hoe groot het totale bestand is   *Door de omvang van een bestand vooraf ter beschikking te stellen kunnen de benodigde resources al vooraf gepland worden.*                                                                                                                                                                                                                                                                                                                                                                                                                              |
| MD006          | De metadata MOET een checksum geven van het bestand (element `<checksum)>`. Indien met deelbestanden wordt gewerkt: de metadata MOET een checksum geven van het totale bestand (element `<checksum>`). Deze checksum dient te worden weergegeven als een string van hexadecimale digits.<br>Toegestane algoritmen zijn: MD5, SHA-1, SHA256, SHA384, SHA512.<br>Aanbevolen algoritmen zijn: SHA256, SHA384, SHA512. <br><br>*Door een checksum toe te voegen kan de inhoud van een bestand na de overdracht geverifieerd worden.*<br>*(De toepassing van deze algoritmen is hier gericht op (en beperkt tot) detectie van transmissiefouten mbv een checksum. MD5 en SHA-1 zijn daarom in deze context en in verband met backwards compatibiliteit opgenomen als toegestaan).*                                                                                   |
| MD007          | De metadata MOET de naam van het bestand opgeven, als string, met een lengte van maximaal 200 karakters (element `<filename>`). De toegestane karakters zijn letters, cijfers, punt, underscore, en hyphen.  De naam van het bestand moet uniek zijn in de context van de uitwisseling tussen twee partijen (OIN verzender – OIN ontvanger).  *De eisen ten aanzien van bestandnamen kunnen voor ieder platform verschillend zijn; daarom kan de opgegeven bestandsnaam niet altijd als bestandsnaam aan de zijde van de ontvanger gebruikt worden.*                                                                                                                                                                                                                                                                                             |
| MD008          | De metadata MAG aangeven wat de context is van het Digikoppeling bericht waar het onderdeel vanuit maakt (attribuut `<contextId>`).  *Met behulp van de contextID is het mogelijk om de context van de applicatie op te nemen. Ook is het mogelijk een correlatie aan te brengen tussen het bestand en de metadata. Daarvoor moet het bestand dezelfde contextID bevatten.*                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| MD009          | De metadata MOET het Internet media type (MIME type of Content-type) specificeren van het bestand (element `<contentType>`) [[rfc2046]].                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| MD010          |  De metadata MOET bij pull de URL van de verzender (element `<senderUrl>` en bij push de URL van de ontvanger (element `<receiverUrl>`) bevatten.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

In de bijlagen zijn de XML Schema definities (XSD) en XML voorbeeldberichten
opgenomen.

## Opbouw en structuur XML Schema definities  


Om volledig backwards compatible te blijven met de vorige versie van de DK GB
standaard waarin alleen het PULL principe was gespecificeerd worden voor PULL en
PUSH aparte schema’s gehanteerd waarbij het PULL schema ongewijzigd is t.o.v. de
vorige versie.

### PULL Schema

Het PULL schema bevat een request bericht definitie. De DK GB PULL variant laat
de invulling van de response verder vrij en definieert alleen (size error) en
(checksum error) in algemene zin.

### PUSH Schema

Het PUSH schema kent een request en een response bericht definitie.

De PUSH request biedt de mogelijkheid om eventueel aan te geven in het bericht
dat het grote bestand uit meerdere onderdelen (‘parts’) bestaat.

Gebruik van de PUSH response definitie is optioneel. Partijen mogen ook
overeenkomen om een eigen berichtstructuur voor de response te gebruiken. Indien
het responsebericht van het PUSH XML Schema wordt gebruikt moet het volledig
worden ingevuld (D.w.z. de ontvangststatus van het grote bestand en van de
eventuele onderdelen zowel bij succesvolle ontvangst als bij fouten).

### PUSH response bericht statuscodes

De volgende statuscodes zijn voorgedefinieerd in het PUSH responsebericht:

- OK

- FILE_NOT_FOUND

- CHECKSUM_TYPE_NOT_SUPPORTED

- CHECKSUM_ERROR

- INCORRECT_FILE_SIZE

- COMPRESSION_NOT_SUPPORTED

- DECOMPRESSION_ERROR

- UNKNOWN_ERROR

Zie ook de XSD’s in de bijlage.

