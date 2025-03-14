# Koppelvlakstandaard WUS

## Inleiding 

WUS is een acroniem voor WSDL, UDDI en SOAP: WUS dus. Daarmee wordt een familie
van internationale standaarden van OASIS en W3C bedoeld; deze worden ook vaak
met WS-\* aangeduid. Deze standaarden gaan over application-to-application
webservices.

Alle Digikoppeling webservices die op WUS gebaseerd zijn, moeten aan deze
koppelvlakstandaard conformeren. Doel van de Koppelvlakstandaard WUS is het
eenduidig en bruikbaar definiëren van het koppelvlak voor WUS. Er zijn daarvoor
de volgende onderwerpen relevant: WSDL met adressering en naamgeving,
beveiliging, betrouwbaarheid, binaire data, resulterende berichtheaders,
profielen van Digikoppeling en compliancevoorzieningen. Deze standaard wordt
verder aangevuld met “Best Practices” en adviezen, beschreven in een apart
document. Alle documentatie over Digikoppeling is te vinden op [[?Logius]].

### WSDL

Een webservice wordt deels formeel en automatisch verwerkbaar gedefinieerd
(beschrijving – description) door een WSDL. Deze WSDL geeft een beschrijving van
de eisen die ten aanzien van de communicatie gesteld worden. Een WSDL kan onder
andere bestaan uit meerdere schema definities in aparte XSD’s en policy
definities. Gezamenlijk vormt dit een abstracte definitie van de webservice. De
webservice communiceert feitelijk door middel van SOAP berichten, die
gegenereerd worden op basis van de WSDL. Adressering en naamgeving zijn
specifieke aandachtsgebieden.

### Resulterende berichtheaders

Deze standaard beschrijft per definitie aan welke eisen een Digikoppeling WUS
implementatie moet voldoen. De praktijk leert dat dergelijke eisen vaak erg
abstract zijn en dus gebaat zijn bij voorbeelden. Voorbeelden van berichten zijn
gepubliceerd op [[?Logius]]. Voorbeelden van WSDL’s zijn beschikbaar als
onderdeel van [[?Compliance]].

## Ondersteunde varianten 

Om redenen van interoperabiliteit, eenvoud en overzichtelijkheid onderscheidt
deze Digikoppeling Koppelvlakstandaard een tweetal varianten van uitwisselingen.
Elke variant veronderstelt bepaalde voorgedefinieerde keuzes voor parameters als
beveiliging en betrouwbaarheid en is daarmee een “profiel” voor WUS.

Elke uitwisseling op basis van de WUS-protocollen over Digikoppeling zal moeten
voldoen aan één of een combinatie van de volgende Digikoppeling WUS-varianten:

- __Best Effort:__ dit zijn synchrone uitwisselingen die geen faciliteiten
    voor betrouwbaarheid (ontvangstbevestigingen, duplicaateliminatie etc.)
    vereisen. Voorbeelden zijn toepassingen waar het eventueel verloren raken
    van sommige berichten niet problematisch is en waar snelle verwerking
    gewenst is.

- __End-to-End Security:__ een bericht wordt beveiligd tussen de
    uiteindelijke consumer en de uiteindelijke provider, ook wanneer er zich
    intermediairs bevinden in het pad tussen die twee. Het betreft hier
    authenticatie van de consumerorganisatie, conform het Digikoppeling
    authenticatiemodel, waarbij alleen de identiteit van de consumerorganisatie
    relevant is(signing), en encryptie van het bericht (payload inclusief
    attachments) onderweg. Voor de authenticatie en encryptie wordt
    gebruikgemaakt van WS-Security.

- __Attachments:__ één of meerdere bijlagen, naast natuurlijk het reeds bestaande
    (xml) bericht zelf. Dit kan, maar hoeft niet, toegepast te worden in
    combinatie de bovengenoemde profielen: het is dus optioneel.

- Vertrouwelijkheid en authenticatie van zender en ontvanger wordt als volgt
    gerealiseerd:

  - voor Point-to-Point Security, door middel van twee-zijdig TLS op
 transport-niveau (in het HTTP kanaal). (De toepassing ervan is verplicht
 op alle Digikoppeling versies.)

  - voor End-to-End Security, door middel van signing (ondertekening) en
 (optioneel) encryptie (versleuteling) op berichtniveau (payload
 inclusief de attachments, ook wel 'bijlagen' genoemd) in combinatie met
 (point-to-point) twee-zijdig TLS in het HTTP kanaal.

De onderstaande tabel geeft in essentie de eigenschappen van de verschillende
Digikoppeling profielen weer. Voor alle profielen wordt tweezijdig TLS gebruikt
op transportniveau (HTTPS).

<table>
<thead>
  <tr>
    <td colspan="3">Profile Names</td>
    <td colspan="4">Profile Names</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="3">Digikoppeling WUS</td>
    <td>2-zijdig TLS</td>
    <td>Signed</td>
    <td>Encrypted</td>
    <td>Attachments</td>
  </tr>
  <tr>
    <td colspan="2">Best Effort</td>
    <td>Digikoppeling 2W-be</td>
    <td>√</td>
    <td>―</td>
    <td>―</td>
    <td>Optional</td>
  </tr>
  <tr>
    <td rowspan="2">End-to-End Security</td>
    <td>Best Effort – Signed</td>
    <td>Digikoppeling 2W-be-S</td>
    <td>√</td>
    <td>√</td>
    <td>―</td>
    <td>Optional</td>
  </tr>
  <tr>
    <td>Best Effort – Encrypted</td>
    <td>Digikoppeling 2W-be-SE</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>Optional</td>
  </tr>
</tbody>
</table>

## Compliancevoorzieningen

Voor de Digikoppeling Koppelvlakstandaard WUS is een compliancevoorzieningen
beschikbaar gesteld, waarmee een ontwikkelaar of beheerder kan testen “of het
berichtenverkeer werkt”. Deze compliancevoorziening is gedefinieerd aan de hand
van de WSDL’s die opvraagbaar zijn op
[[?Compliance]].

De volgende interacties zijn beschikbaar:

- Digikoppeling service requester: wordt gebruikt om een ontwikkelde servicere
    requester (client) te testen.

- Digikoppeling service provider: wordt gebruikt om een ontwikkelde service
    provider (service) te testen.

De documenten over de compliancevoorzieningen zijn te vinden op [[?Logius]]. De Digikoppeling Compliancevoorziening zelf is beschikbaar op
[[?Compliance]]

## Gehanteerde standaarden

In dit hoofdstuk worden de verschillende versies van de onderliggende
internationale standaarden vermeld.

### Overwegingen

Primair wordt gekozen voor de interoperabele profielen van OASIS WS-BRSP
(voorheen WS-I). Het gaat dan om WS-I Basic Profile (BP) 1.2, een set
specificaties van webservices die interoperabiliteit bevorderen. Digikoppeling
kiest voor standaarden die algemeen interoperabel beschikbaar zijn, dat wil
zeggen interoperabel geïmplementeerd zijn in het grootste deel van de
(ontwikkel) tools. De kans daarop is groter bij “final” standaarden dan bij
drafts. Digikoppeling kiest daarom voor WS-I Standaarden met status final.

De minimaal ondersteunde TLS encryptie algoritmen en sleutellengtes worden
beschreven in het [[Beveiligingsdocument]].

Resulterende beslissingen ten aanzien van standaarden:

- WS-I BP 1.2; deze is gebaseerd op onderliggende standaarden: SOAP 1.1, WSDL
    1.1 en de te kiezen onderdelen van WS-Addressing en MTOM.

- WS-I BSP 1.1 voor berichtbeveiliging op basis van WS-Security.

- De TLS versies zoals beschreven in [[Beveiligingsdocument]],
    conform de aanbevelingen in WS-I BSP 1.0 voor beveiliging op
    transport/kanaal niveau en aanbevelingen van NIST en NCSC.

- WS-I Simple SOAP Binding Profile Version 1.0.

Bovenstaande standaarden zijn gebaseerd op diverse onderliggende standaarden.
Gevolg is dat de Digikoppeling WUS standaard gebruikmaakt van de volgende set
van standaarden:

| Standaarden | Gevolg van onder andere |
|---|---|
| HyperText Transfer Protocol 1.1 (RFC7230 t/m RFC7233) | WS-I Basic Profile 1.2 |
| SOAP 1.1 | WS-I Basic Profile 1.2 |
| WSDL 1.1 [[wsdl]]| WS-I Basic Profile 1.2 |
| XML 1.0 (Second Edition) | WS-I Basic Profile 1.2 |
| XML Schema Part 1: Structures | WS-I Basic Profile 1.2 |
| XML Schema Part 2: Data types | WS-I Basic Profile 1.2 |
| De huidig toegestane TLS versies zoals beschreven in [[Beveiligingsdocument]] | WS-I Basic Profile 1.2,  NCSC, NIST, ENISA |
| HTTP over TLS Transport Layer Security (RFC2818) | WS-I Basic Profile 1.2 |
| Internet X.509 Public Key Infrastructure Certificate and CRL Profile (RFC 3280) | PKI overheid 1.1 |
| WS-Addressing 1.0 [[ws-addr-metadata]]| WS-I Basic Profile1.2 |
| Web Services Addressing 1.0 – Metadata | WS-I Basic Profile 1.2 |
| SOAP 1.1 Binding for MTOM 1.0 | WS-I Basic Profile 1.2 |
| WS-Security 1.1 | WS-I Basic Security Profile 1.1 |
| WS-Security 1.0 | WS-Security 1.1 |

## Voorschriften 

De inhoud van de Digikoppeling standaard vloeit voort uit bovengenoemde
standaarden, of vormt een nadere invulling daarvan (een keuze uit toegestane
mogelijkheden in die standaarden).

In de paragrafen hierna worden die onderdelen achtereenvolgens behandeld:

- WSDL,

- WS-Addressing,

- Binaire data,

- Beveiliging,

- Betrouwbaarheid,

- Namespace.

### Service definitie WSDL

Voorschriften ten gevolge van de keuze voor BP 1.2

<!-- pieter: deze tabel in plain html, omdat het html voorbeeld niet goed rendert met markdown tabellen -->
<table class="dkkvs">
    <thead>
 <tr>
 <th>Nr</th>
 <th>Omschrijving</th>
 </tr>
    </thead>
    <tbody>
 <tr>
 <td>WW001</td>
 <td>Voor de SOAP berichten wordt SOAP 1.1 en “document-literal binding” gehanteerd. Hierbij wordt als
 transport binding HTTP voorgeschreven.</td>
 </tr>
 <tr>
 <td>WW002</td>
 <td>Door het opleggen van het SOAP style type “document/literal” zal de inhoud van de berichten beschreven
 worden door XML en geen afgeleide daarvan. Dit houdt in dat er niet een eigen mapping mag worden
 geïntroduceerd voor encoding types zoals bijvoorbeeld bij SOAP encoding het geval is. Kortom, de
 datatypen moeten voldoen aan de XML Schema Part 2: Datatypes</td>
 </tr>
 <tr>
 <td>WW003</td>
 <td>Bij document –literal mag het SOAP “body” element slechts 1 XML element bevatten. Hierbinnen kunnen
 eventueel wel meerdere elementen opgenomen worden.</td>
 </tr>
 <tr>
 <td>WW004</td>
 <td>Basic Profile stelt eisen aan het “PortType” van een WSDL. Hierbij mogen de “parts” van de “messages”
 alleen een “element” bevatten (geen “parts” die een “type” attribuut gebruiken). “R2204 A
 document-literal binding in a <strong>DESCRIPTION</strong> MUST refer, in each of its soapbind:body
 element(s), only to wsdl:part element(s) that have been defined using the element attribute.” Er is geen
 voorbeeld bij WS-I, maar een voorbeeld kan zijn:<pre class="example">
 &lt;element name="TradePriceRequest"&gt;
 &lt;complextype\&gt;
 &lt;all&gt;
 &lt;element name="tickerSymbol" type="string"&gt;&lt;/element&gt;
 &lt;/all&gt;
 &lt;/complextype\&gt;
 &lt;/element&gt;
 …
 &lt;message name="GetLastTradePriceInput"&gt;
 &lt;part name="body" elements="xsd1:TradePriceRequest"&gt;&lt;/part&gt;
 &lt;/message&gt;
 </pre></td>
 </tr>
    </tbody>
</table>

Aanvullende voorschriften (dus specifieke Digikoppeling-invulling binnen de ruimte van een bovengenoemde standaard)


| Nr    | Omschrijving |
|---|---|
| WS001 | Er kunnen meerdere operaties per webservice gedefinieerd worden. |
| WS002 | De SOAPAction aanduiding in de WSDL wordt gevuld met een lege string (“”), wordt weggelaten, of heeft dezelfde vulling als de { [http://www.w3.org/2007/05/addressing/metadata](http://www.w3.org/2007/05/addressing/metadata)}Action in de WSDL. In de HTTP Header van het bericht moet de SOAPAction een lege string met quotes zijn (“”), of een waarde hebben gelijk aan de WS-Addressing Action (wsa:Action). |
| WS003 | De Digikoppeling WUS ondersteunt alleen de zogenaamde “request/response” berichtenuitwisseling (zie WSDL 1.1 specificatie paragraaf “2.4 Port Types”) [[wsdl]]. |
| WS005 | De WSDL bevat slechts één “portType” per WSDL bestand. |
| WS006 | Digikoppeling ondersteunt alleen UTF-8. |
| WS007 | In de header zijn geen eigen velden (header blocks) toegestaan. De header bevat alleen de in het betreffende profiel vastgestelde velden, die dus uitsluitend gedefinieerd zijn in het betreffende WS-I profiel (respectievelijk de onderliggende OASIS/W3C standaarden). |
| WS008 | Het is verplicht een WS-Addressing Action referentie op te nemen in de WSDL. Het definiëren van een WS-Addressing action in WSDL kan met behulp van de Web Services Addressing 1.0 – Metadata standaard. Informatie hierover is te vinden via [http://www.w3.org/TR/2007/REC-ws-addr-metadata-20070904/\#explicitaction](http://www.w3.org/TR/2007/REC-ws-addr-metadata-20070904/#explicitaction) [[ws-addr-metadata]]. Zie voor mogelijke vulling van wsam:action in WSDL “4.4.4 Default Action Pattern for WSDL 1.1” van de Web Services Addressing 1.0 – Metadata standaard (http://www.w3.org/TR/2007/REC-ws-addr-metadata-20070904/). |

### WS-Addressing

Voorschriften als gevolg van het toepassen van WS-Addressing
<!-- pieter: deze tabel in plain html, omdat het html voorbeeld niet goed rendert met markdown tabellen -->
<table class="dkkvs">
    <thead>
        <tr>
            <th>Nr</th>
            <th>Omschrijving</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>WA001</td>
            <td>Digikoppeling WUS gebruikt de volgende velden uit
                WS-Addressing:<br>
                <ul>
                    <li>wsa:To</li>
                    <li>wsa:Action</li>
                    <li>wsa:MessageID</li>
                    <li>wsa:RelatesTo</li>
                    <li>wsa:ReplyTo</li>
                    <li>wsa:From</li>
                </ul>
                De communicatie binnen het Digikoppeling domein is voor een deel afhankelijk van de toepassing van
                WS-Addressing velden. Aangezien er meerdere WS-Addressing specificaties zijn, die onder meer
                verschillende namespaces kunnen hebben, is er voor gekozen om alleen de specificatie van 2006/05
                [https://www.w3.org/TR/ws-addr-core/](https://www.w3.org/TR/ws-addr-core/)[[ws-addr-core]] verplicht te stellen in de berichten
                binnen het Digikoppeling domein. Hieronder wordt de toepassing van de verschillende velden toegelicht.
                Er is gekozen voor een zo klein mogelijke subset uit de WS-Addressing standaard om de kans op
                interoperabiliteitsissues te minimaliseren. Met het toepassen van deze standaard wordt het “achter de voordeur” routeren mogelijk.<br><br>
                <strong>wsa:To</strong> Dit wordt gebruikt om de endpoint vast te leggen waar het bericht naar toe dient te gaan. Het element wsa:to is van het type wsa:AttributedURIType - een extensie op het xs:anyUri type- en dient gevuld te worden met een ‘Adres’ element. De waarde van het adres element kan hetzij een absolute URI zijn of `http://www.w3.org/2005/08/addressing/anonymous`.
                Optioneel kan het To-adres aangevuld te worden met een OIN door het gebruik van querystring parameters (bijvoorbeeld `http://service-end-point?OIN=xxxxxx`). De waarde van de OIN in het adres is het OIN nummer van de ontvangende partij.<br><br>
                <strong>wsa:Action</strong> Deze waarde wordt gebruikt om een specifieke operatie aan te roepen. Deze waarde is terug te vinden in de WSDL van de betreffende aan te roepen webservice van de Service Provider. Dit veld is verplicht en moet in het bericht worden opgenomen.<br><br>
                <strong>wsa:MessageID</strong> De waarde hiervan kan door de service requester of provider zelf ingevuld worden zolang dit een waarde is die aan de onderliggende specificatie voldoet ([[ws-addr-core]]).<br><br>
                <strong>wsa:RelatesTo</strong> Dit element komt alleen voor in de SOAP header van het response bericht. Het bevat de waarde van de wsa: MessageID van het request bericht.<br><br>
                <strong>wsa: ReplyTo</strong> De verplichte specificatie van wsa:ReplyTo geldt alleen voor het request bericht. De specificatie mag zowel plaatsvinden door gebruik te maken van de default-waarde als door expliciete opname van deze SOAP-header. Voor synchrone communicatie t.b.v. bevragingen zal het replyTo veld gevuld zijn met de waarde `http://www.w3.org/2005/08/addressing/anonymous` of het element volledig weglaten. Bij weglaten van dit veld moet de ontvanger conform WS-Addressing specificatie alsnog de waarde `http://www.w3.org/2005/08/addressing/anonymous` gebruiken.<br><br>
                <strong>wsa:From</strong> Het gebruik van wsa:From is optioneel voor synchrone berichten voor
                bevragingen. De waarde van dit veld wordt gebruikt om aan te geven waar het bericht vandaan komt. De wsa:From is van het type wsa:EndPointReferenceType en dient gevuld te worden met een ‘Adres’ element (wsa:Address). De waarde van het adres element kan hetzij een absolute URI zijn of `http://www.w3.org/2005/08/addressing/anonymous`. Optioneel kan het From-adres aangevuld te worden met een OIN door het gebruik van querystring parameters (e.g. `http://service-end-point?OIN=xxxxxx`). De waarde van de OIN in het adres is het OIN nummer van de verzendende partij.<br><br>
                De elementen <strong>wsa:ReplyTo</strong> en <strong>wsa:From</strong> zijn beiden van de type
                ‘wsa:EndPointReferenceType’. Het EndPointReferenceType stelt enkel het element ‘Address’ verplicht. De overige velden van EndPointReferenceType zijn optioneel en zijn om compatibiteitsredenen niet toegestaan binnen Digikoppeling.<br><br>
                Het is toegestaan om overige WS-Addressing velden op te nemen in de berichten omdat bij sommige toolkits het genereren van deze velden niet onderdrukt kan worden. Hierbij geldt wel de beperking dat de waarde voor deze velden het routeringsmechanisme niet verstoort. Derhalve moet, indien het bericht andere velden dan hierboven bevat, de waarde `http://www.w3.org/2005/08/addressing/anonymous` of `http://www.w3.org/2005/08/addressing/none` aan deze velden toegekend worden. Overzicht verplichte WS-Addressing properties in request en response berichten (volgens
                [[ws-addr-metadata]])
                <br><br>
                <b>WS-Addressing request headers Field</b>
                <table class="dkkvsincell">
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>Property</th>
                            <th>Mandatory</th>
                            <th>Description.</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>wsa:To</td>
                            <td>[destination]</td>
                            <td>Y</td>
                            <td>Provides the address of the intended receiver of this message.</td>
                        </tr>
                        <tr>
                            <td>wsa:Action</td>
                            <td>[action]</td>
                            <td>Y</td>
                            <td>Identifies the semantics implied by this message.</td>
                        </tr>
                        <tr>
                            <td>wsa:From</td>
                            <td>[source endpoint]</td>
                            <td>Y<sup>1</sup></td>
                            <td>Provides the address of the original sender of this Message</td>
                        </tr>
                        <tr>
                            <td>wsa:ReplyTo</td>
                            <td>[reply endpoint]</td>
                            <td>Y<sup>2</sup></td>
                            <td>Intended receiver for the reply to this message.</td>
                        </tr>
                        <tr>
                            <td>wsa:FaultTo</td>
                            <td>[fault endpoint]</td>
                            <td>N</td>
                            <td>Intended receiver for faults related to this message. May be included to direct fault
                                messages to a different endpoint than [reply endpoint]</td>
                        </tr>
                        <tr>
                            <td>wsa:MessageID</td>
                            <td>[message id]</td>
                            <td>Y</td>
                            <td>Unique identifier for this message. Used in the [relationship] property of the reply
                                message. </td>
                        </tr>
                        <tr>
                            <td>wsa:RelatesTo</td>
                            <td>[relationship]</td>
                            <td>N</td>
                            <td>Indicates relationship to a prior message. Unused in this Message Exchange Pattern
                                (MEP), but could be included to facilitate longer running message exchanges.</td>
                        </tr>
                    </tbody>
                </table>
                <sup>1</sup>Voor bevragingen is source endpoint optioneel.<br>
                <sup>2</sup>Impliciet specificeren van het reply endpoint door weglaten van ReplyTo is
    ook toegestaan.
                <br><br>
                <b>WS-Addressing response headers</b>
                <table class="dkkvsincell">
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>Property</th>
                            <th>Mandatory</th>
                            <th>Description.</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>wsa:To</td>
                            <td>[destination]</td>
                            <td>N<sup>3</sup></td>
                            <td>Provides the address of the intended receiver of this message.</td>
                        </tr>
                        <tr>
                            <td>wsa:Action</td>
                            <td>[action]</td>
                            <td>Y</td>
                            <td>Identifies the semantics implied by this message.</td>
                        </tr>
                        <tr>
                            <td>wsa:From</td>
                            <td>[source endpoint]</td>
                            <td>N</td>
                            <td>Message origin. Unused in this MEP, but could be included to facilitate longer running message
                                exchanges.</td>
                        </tr>
                        <tr>
                            <td>wsa:ReplyTo</td>
                            <td>[reply endpoint]</td>
                            <td>N</td>
                            <td>Intended receiver for replies to this message. Unused in this MEP, but could be included to facilitate
                                longer running message exchanges.</td>
                        </tr>
                        <tr>
                            <td>wsa:FaultTo</td>
                            <td>[fault endpoint]</td>
                            <td>N</td>
                            <td>Intended receiver for faults related to this message. Unused in this MEP, but could be included to
                                facilitate longer running message exchanges.</td>
                        </tr>
                        <tr>
                            <td>wsa:MessageID</td>
                            <td>[message id]</td>
                            <td>Y<sup>4</sup></td>
                            <td>Unique identifier for this message. Unused in this MEP, but may be included to facilitate longer running
                                message exchanges.</td>
                        </tr>
                        <tr>
                            <td>wsa:RelatesTo</td>
                            <td>[relationship]</td>
                            <td>Y</td>
                            <td>Indicates that this message is a reply to the request message, using the request message [message id] value and the predefined <a
                                    href="http://www.w3.org/2005/08/addressing/reply">http://www.w3.org/2005/08/addressing/reply</a>
                                IRI.</td>
                        </tr>
                    </tbody>
                </table>
                <sup>3</sup>Sommige platformen wijken op dit punt af van de Web Service Addresing 1.0 –
Metadata standaard. Het wsa:To veld wordt bij synchrone SOAP verkeer actief uit
het antwoordbericht gefilterd. Om hier vanuit de standaard aan tegemoet te komen
mag bij het ontbreken van dit veld in het antwoordbericht door de ontvanger de
anonymous waarde (http://www.w3.org/2005/08/addressing/anonymous) worden
aangenomen.

<sup>4</sup> Hiermee wordt afgeweken van wat de Web Services Addressing 1.0 – Metadata
standaard voorschrijft. Volgens deze standaard is de MessageID in response
optioneel. Bovenstaande properties kunnen in een aantal gevallen ook gespecificeerd worden
door betreffende velden in de header weg te laten. Zie WS-addressing 1.0- Core, paragraaf 2.1 en paragraaf 3.2; zie ook BP 1.2 paragraaf 3.7.14.
            </td>
        </tr>
    </tbody>
</table>

### Binaire Data

In de Digikoppeling Koppelvlakstandaard WUS worden twee mogelijkheden
ondersteund om binaire data te versturen. Dat zijn Base64Binary (Base64Binary in
payload element van het bericht) of MTOM (MIME wrappers waarbij binaire data in
een aparte Multipart/Related pakket is opgenomen). Bij het toepassen van MTOM
wordt er ook wel gesproken van een geoptimaliseerd bericht.

De meest gangbare toolkits kunnen MTOM berichten ontvangen en versturen. Het wel
of niet toepassen van MTOM kan vaak vanuit de code of middels een
configuratiebestand geregeld worden.

Bij het inrichten bepaalt de provider of een koppelvlak wel of geen
ondersteuning biedt voor MTOM. Bij een nieuwe koppeling in samenspraak, bij
toevoegen van een afnemer aan een bestaande dienst dient deze zich te
conformeren aan de bestaande inrichting (en wel of niet gebruik van MTOM).

| Nr    | Omschrijving |
|---|---|
| WM001 | Toepassen MTOM wordt door webservice provider bepaald.  |

### Beveiliging

Point-to-Point en End-to-End beveiliging wordt ondersteund. Point-to-Point
beveiliging wordt uitgevoerd op basis van TLS, End-to-End beveiliging op basis
van WS-Security.

#### Point-to-Point beveiliging

Deze beveiliging zorgt ervoor dat het volledige bericht en het http-protocol is
beveiligd tijdens het transport van verzender naar ontvanger. Alle Digikoppeling
profielen verplichten point-to-point beveiliging. Hierbij gelden de volgende
voorschriften:

| Nr    | Omschrijving |
|---|---|
| WT001 | Authenticatie op transportniveau gebeurt op basis TLS met tweezijdige authenticatie. De huidige toegestane protocolversies zijn beschreven in het [[Beveiligingsdocument]]. Client and Server authenticatie is vereist gebruikmakend van HTTPS en alle in [[Beveiligingsdocument]] genoemde TLS versies. De TLS implementatie mag niet op een oudere TLS of SSL versie terug kunnen vallen. |
| | Meer informatie in het [[Beveiligingsdocument]] |
| WT002 | De te gebruiken certificaten in de productie omgeving voldoen aan de eisen van PKIoverheid (PvE 3b) en de inhoud van de identificerende velden in het certificaat dienen te voldoen aan de afspraken als gesteld in de functionele eisen Authenticatie Digikoppeling. Met het toepassen van PKIoverheid-certificaten die Digikoppeling compliant zijn, wordt hieraan voldaan. |
| WT003 | De minimaal ondersteunde TLS encryptie algoritmen en sleutellengtes worden beschreven in het [[Beveiligingsdocument]] |
| | Meer informatie in het [[Beveiligingsdocument]] |
| WT004 | De geldigheid van het certificaat wordt getoetst met betrekking tot de geldigheidsdatum en de Certificate Revocation List(CRL) die voldoet aan de eisen van PKIoverheid. |
| WT005 | De betreffende CRL dient zowel voor de versturende als ontvangende partij te benaderen zijn. |
| WT006 | Voor communicatie over HTTPS wordt port 443 gebruikt. |
| | Overwegingen: Wanneer men afwijkt van Poort 443 dient de gebruiker van de site of de service naast https ook het afwijkende poortnummer in de URI te specificeren. Het is sterk aanbevolen voor publieke services en sites om poort 443 te handhaven en met behulp van een firewall rule of proxy pass het verkeer intern te redirecten naar een afwijkende poort. Het verbergen van een open poort door een afwijkend poortnummer te gebruiken heeft geen zin omdat port scans eenvoudig open en toegankelijke poorten ontdekken. |
| WT007 | Binnen een TLS-sessie kunnen meerdere berichten verstuurd worden. |
| WT008 | Voor de TLS-sessie moet een maximale duur gelden, na het verloop hiervan wordt de verbinding verbroken. Partijen zijn vrij om de maximale duur zelf te bepalen. |

#### End-to-End beveiliging

Deze beveiliging is optioneel en wordt bovenop point-to-point beveiliging
ingezet op SOAP niveau met behulp van ondertekening en versleuteling. End-to-End
beveiliging is primair van toepassing in de scenario’s waar intermediairs
betrokken zijn gedurende de gegevensuitwisseling en in scenario’s waarbij
onweerlegbaarheid van belang is.

| Nr | Omschrijving | 
|---|---|
| WB001 | Toepassen WS-Security 1.0 en WS-Security 1.1 | 
| | Overwegingen: Basic Security Profile 1.1 is sinds 2010 november final geworden. Hierin worden zowel de WS-Security 1.0 als de WS-Security 1.1 namespaces beide gebruikt. |
| WB002 | Toepassen van Timestamp in security header met Timestamp Created is verplicht. Timestamp Expires is optioneel.<br>De tijdstamp moet een Universal Time Coordinated (UTC) tijdzone aanduiding hebben. Bij het toepassen van een timestamp gaat tijdsynchronisatie van de verschillende communicerende systemen een rol spelen. Indien dit niet mogelijk is moet hiermee met de vulling van de Created en Expires rekening worden gehouden door middel van een “timestampSkew“. | 
| | Overwegingen: Bij toepassen van Timestamp Expires is tijdsynchronisatie van belang. Om mogelijke problemen hiermee te voorkomen, zou er overwogen kunnen worden om een eis op te nemen dat de Expires niet in Timestamp opgenomen mag worden. Omdat het expliciet weglaten van de Expires niet in alle tooling mogelijk is, wordt hiervoor niet gekozen. Tevens kan het zijn dat door het ontbreken van tijdsynchronisatie er problemen zijn met de Timestamp Created, in de situatie waarbij de ontvanger heeft vastgesteld dat de Timestamp Created in de toekomst ligt. Hiervoor biedt tooling vaak een “timestampSkew”. Deze geeft de toegestane afwijking ten opzichte van UTC aan.  |
| WB003 | Indien WS-Security wordt toegepast, is ondertekenen verplicht en versleutelen optioneel (keuze profiel Digikoppeling 2W-be-S, Digikoppeling 2W-be-SE,). |
| | Overwegingen: De berichten kunnen zowel ondertekend als versleuteld worden. Gezien het doel van WS-Security, te weten het “door een intermediair heen” kunnen doorgeven van authenticatie-informatie, is ondertekenen primair van belang; daarmee is ook onweerlegbaarheid geregeld. Uiteraard kan het in een bepaalde situatie ook een eis zijn dat het bericht niet leesbaar is voor de intermediair. | 
| WB004 | Ondertekenen van bericht onderdelen SOAP:body, SOAP:headers (WS-Addressing headers en Timestamp) is verplicht bij toepassing van End-to-End beveiliging. Van elk van deze onderdelen dient separaat een digest te worden berekend en te worden opgenomen in het SignedInfo element. De handtekening dient te worden gegenereerd op basis van de inhoud van het SignedInfo element. |
| | Overwegingen: Met het ondertekenen wordt authenticatie, integriteit en onweerlegbaarheid ondersteund. Het is van belang dat de integriteit en onweerlegbaarheid van de inhoud en header van het bericht kan worden vastgesteld, de SOAP:body, SOAP:header (WS-Addressing en het Timestamp element) zullen dus ondertekend moeten worden. |
| WB005 | Bij toepassen van versleutelen geldt dit voor de volgende bericht onderdelen: SOAP:body | |
| | Overwegingen: De WS-Addressing headers worden niet versleuteld, dit omdat deze anders niet leesbaar zijn voor intermediairs. De wsa:Action en de ondertekening worden dus niet versleuteld. Ook de SOAPaction kan niet afgeschermd worden. Door beveiliging op transport niveau is het risico van een niet afgeschermde wsa:Action en SOAPAction tot een minimum beperkt. |
| WB006 | Berichten worden eerst ondertekend en vervolgens versleuteld. | 
| | Overwegingen: Omdat er zowel wordt ondertekend als versleuteld, moet de volgorde hiervan gespecificeerd worden: -Bij verzending eerst ondertekenen en vervolgens versleutelen. -Bij ontvangst eerst ontsleutelen en daarna de ondertekening verifieren. |


<table class="dkkvs">
    <tbody>
        <tr>
            <td>WB007</td>
            <td>Technische gegevens ten behoeve van ondertekenen
                <table class="dkkvsincell">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Waarde</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Onderteken optie Algoritme</td>
                            <td>Exclusive XML Canonicalization
                                [http://www.w3.org/2001/10/xml-exc-c14n](http://www.w3.org/2001/10/xml-exc-c14n\#)
                        </tr>
                        <tr>
                            <td>Data Encryption Algorithms</td>
                            <td>zie [[Beveiligingsdocument]]
                        </tr>
                        <tr>
                            <td>Key Transport Algorithms</td>
                            <td>zie [[Beveiligingsdocument]]
                    </tbody>
                </table>
            </td>
        </tr>
        <tr>
            <td>WB008</td>
            <td>Technische gegevens ten behoeve van versleutelen:
                <table class="dkkvsincell">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Waarde</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Data Encryption Algorithms</td>
                            <td>zie [[Beveiligingsdocument]]
                        </tr>
                        <tr>
                            <td>Key Transport Algorithms</td>
                            <td>zie [[Beveiligingsdocument]]
                    </tbody>
                </table>
            </td>
        </tr>
    </tbody>
</table>

||| 
|---|---|
|WB009|Security token X.509 Certificate Token (PKI Overheid Digikoppeling certificaat).|
|WB010|Publieke sleutel welke gebruikt is voor het signing proces dient meegeleverd te worden met het bericht via een ‘Direct security token’ reference.|
||Overwegingen:<br>Het certificaat wordt in het bericht meegestuurd. Hiermee kan de ontvanger door middel van het meegeleverd certificaat de handtekening controleren. Het certificaat dient uiteraard wel vertrouwd te zijn via een truststore configuratie waarin het PKIoverheid stamcertificaat alsmede de intermediair certificaten en Trusted Servicer Provider certificaten zijn opgenomen. Zie hiervoor [[Certificaten]]. (een vereiste voor veel platformen om de validatie van het bericht aan te vangen).|
|WB011|Het toepassen van End-to-End beveiliging wordt op serviceniveau aangeduid. Alle operaties en dus berichten (request en response) worden ontsloten volgens één bepaald Digikoppeling profiel.|
||Overwegingen:<br>Beveiligingseisen kunnen op het niveau van het bericht gedefinieerd worden, maar niet alle toolkits kunnen hiermee overweg. Totdat alle belangrijke toolkits dit wel kunnen, is het beter om bericht beveiliging op serviceniveau te definiëren.|
|WB012|Voor het versleutelen van het responsebericht wordt het certificaat in het requestbericht gebruikt.|
||Toelichting:<br>In eis WB010 wordt aangegeven dat het certificaat voor ondertekening in het bericht wordt opgenomen. Indien een webservice wordt ontsloten volgens het Digikoppeling 2W-be-SE profiel moet deze op basis van het requestbericht kunnen bepalen welk certificaat gebruikt moet worden om de payload van de response te versleutelen. Dit kan door het certificaat in het requestbericht te gebruiken voor versleuteling van de response. Ook de requester dient hier dus rekening mee te houden bij ontsleutelen van het responsebericht.<br>Om het request bericht initieel te versleutelen dient de publieke sleutel van de ontvangende partij al in de truststore geregistreerd te zijn.|
|WB013|Indien WS-Security wordt toegepast, is het controleren van de signature door de ontvangende partij verplicht.|
||Overwegingen:<br>Het ondertekenen van berichten is alleen zinvol als de ontvanger van het bericht ook daadwerkelijk de signature valideerd. Indien de validatie mislukt, dient het bericht afgewezen te worden en een foutmelding als antwoord te worden verstuurd.<br>Ook indien de ondertekening van de respons niet valide is mogen de gegevens niet verwerkt worden. De ontvanger kan wedermaal een requestbericht versturen, maar de kans is groot dat out-of-band communicatie noodzakelijk is om er voor te zorgen dat de dienstaanbieder een valide respons stuurt.|
|WB014|Indien WS-Security wordt toegepast dient het responsebericht de signature van het requestbericht als onderdeel van het SignatureConfirmation element op te nemen (WS Security 1.1.).|
||Overwegingen:<br>Door het herhalen van de ondertekening van het requestbericht kan de ontvanger van het responsebericht valideren dat het oorspronkelijke requestbericht in onaangetaste staat is ontvangen en verwerkt.<br>Een contract wordt voor een Digikoppeling WUS Koppelvlak gedefinieerd door een WSDL. De WSDL 1.1 specificatie op zich biedt geen mogelijkheden om het gebruik van WS-Security aan te geven.|

### Naamgeving, namespaces

Er worden geen aanvullende eisen gesteld aan de namespaces. In de header mogen
alleen velden voorkomen conform OASIS/W3C standaarden (gebaseerd op de daar
gedefinieerde namespaces), of service-specifieke namen, gedefinieerd door de
provider met een namespace gedefinieerd door de provider. Ook aan de naamgeving
van services, operations etc. worden vanuit Digikoppeling geen eisen gesteld. De
prefix “Digikoppeling” is gereserveerd voor mogelijk toekomstig gebruik.

## Foutafhandeling

Er worden momenteel nog geen eisen gesteld aan de foutafhandeling, anders dan in
de onderliggende specificaties vermeld worden, zoals WS-I Basic Profile 1.2. In
het document ” Digikoppeling Best Practices WUS” is een aantal adviezen
opgenomen.

Aanvullende standaarden die gebruikt worden in combinatie met Digikoppeling
kunnen overigens wel eisen stellen aan de fouthandeling.

