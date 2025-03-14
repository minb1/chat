# Koppelvlakstandaard ebMS2

## Inleiding

Dit document specificeert de Koppelvlakstandaard ebMS2 voor
berichtenuitwisseling over Digikoppeling (voorheen OverheidsServiceBus) als een
toepassing van de EBXML-MSG standaard, de ebXML Message Service Specification
versie 2.0 [[EBXML-MSG]]. Digikoppeling is bedoeld als generieke infrastructuur
voor een grote variëteit aan diensten. Deze Standaard is daardoor eveneens
generiek en dient nader gespecialiseerd te worden voor specifieke berichtstromen
en diensten.

EbXML Messaging [[EBXML-MSG]] is bedoeld voor verschillende toepassingen en
faciliteert die diversiteit door een scala aan configureerbare features en
opties te bieden. Elk gebruik van ebXML Messaging in een bepaalde keten of
binnen een bepaalde gemeenschap vereist in de praktijk een bepaalde mate van
aanvullende standaardisatie. Aangezien veel van de configuratiefeatures in de
standaard optioneel zijn, moet precies gedocumenteerd worden welke onderdelen
ervan op welke manier toegepast zijn, om op de verschillende relevante niveaus
interoperabiliteit te realiseren. Die informatie is hier verzameld en
gepubliceerd als configuratiegids voor de gebruikers van Digikoppeling. Het legt
de overeengekomen conventies vast voor het gebruik van ebXML message service
handlers, de functionaliteit die van een implementatie verwacht wordt en de
details voor het gebruik van de standaard.

Een deployment specificatie is niet hetzelfde als een ebXML
samenwerkingsprotocol overeenkomst (ook wel aangeduid met een “Collaboration
Protocol Profile and Agreement) [ISO 15000-1]. Wel hebben sommige onderdelen van
een deployment specificatie gevolgen voor de specifieke invulling van
CPA-elementen.

## Terminologie in dit document

Dit document biedt organisaties die gebruik gaan maken van Digikoppeling de
basis voor de configuratie van de ebXML Messaging software. Een correcte
configuratie is van belang voor het uitwisselen van berichten. Mocht er voor een
bepaald onderdeel geen specifieke richtlijn gegeven zijn, dan wordt dit
aangegeven met één van de volgende waardes:

- **Not applicable**: Dit is voor onderdelen die niet relevant zijn voor
 Digikoppeling, of voor mogelijkheden die niet gebruikt worden.

- **No Recommendation**: geeft aan dat er geen wijziging of voorkeur voor een
 bepaalde invulling van het onderdeel is op het algemene niveau waar dit
 document zich op richt. Specifieke toepassingen van deze specificatie (voor
 specifieke berichtstromen) zullen hier in sommige gevallen wel nog
 aanvullende eisen voor stellen.

- **Pending**: voor onderdelen die nog nader onderzocht worden en mogelijk in
 toekomstige versies nader uitgewerkt worden.

## Ondersteunde varianten

De ebXML Messaging 2.0-standaard is de basis van deze specificatie. Deze
standaard biedt een hogere mate van configureerbaarheid dan in
Digikoppeling-praktijk wenselijk is. Om redenen van interoperabiliteit, eenvoud
en overzichtelijkheid onderscheidt deze koppelvlakstandaard een drietal
varianten van uitwisselingen. Elke variant veronderstelt bepaalde
voorgedefinieerde keuzen voor parameters als synchroniciteit, beveiliging en
betrouwbaarheid en is daarmee een “profiel” voor ebXML Messaging.

Elke uitwisseling op basis van het ebXML Messaging versie 2.0 protocol over
Digikoppeling zal moeten voldoen aan één van de volgende Digikoppeling ebMS2
profielen:

- Best Effort: dit zijn asynchrone uitwisselingen die geen faciliteiten voor
 betrouwbaarheid (ontvangstbevestigingen, duplicaateliminatie etc.) vereisen.
 Voorbeelden zijn toepassingen waar het eventueel verloren raken van sommige
 berichten niet problematisch is en waar snelle verwerking gewenst is.

- Reliable Messaging: asynchrone uitwisseling met ontvangst bevestigingen en
 duplicaateliminatie door de ontvangende message handler\*. Dit profiel is
 onder meer geschikt voor alle berichtenstromen die leiden tot updates van
 gegevensverzamelingen.

 *: In bepaalde gevallen mag een acknowledgement synchroon verstuurd
 worden. Zie par 4.4

- End-to-End Security: op basis van Reliable Messaging of Best Effort wordt
 een bericht beveiligd tussen de uiteindelijke Consumer en de uiteindelijke
 Provider, ook wanneer er zich intermediairs bevinden in het pad tussen die
 twee. Het betreft hier authenticatie van de Consumer organisatie, conform
 het Digikoppeling authenticatiemodel, waarbij alleen de identiteit van de
 Consumerorganisatie relevant is, en encryptie van het bericht onderweg. Voor
 de authenticatie en encryptie wordt gebruik gemaakt van XML digitale
 handtekening [[xmldsig-core-20020212]] en XML-versleuteling [[xmlenc-core]], conform ebMS2.0.

Voor alle profielen gelden de volgende eigenschappen:

- Vertrouwelijkheid en authenticatie van zender en ontvanger wordt als volgt
 gerealiseerd:

- Voor Point-to-Point Security, door middel van twee-zijdig TLS op
 transport-niveau (in het HTTP kanaal). (De toepassing ervan wordt dus ook
 verplicht verklaard bij gebruik van security op berichtniveau.)

- Voor End-to-End Security, door middel van signing (ondertekening) en
 (optioneel) encryptie (versleuteling) op bericht-niveau in combinatie met
 (point-to-point) twee-zijdig TLS in het HTTP kanaal.

- De berichtenuitwisseling is *in principe* asynchroon: een business request
 wordt in een eigen synchrone HTTP request/response sessie verzonden, terwijl
 de acknowledgement en optionele business response via een separaat HTTP
 request/response sessie verzonden worden. In bepaalde gevallen (zie 4.4) mag
 een acknowledgement of een error synchroon verstuurd worden,
 Business*responses* worden altijd asynchroon, in een separaat HTTP sessie
 verzonden.

De onderstaande tabel geeft in essentie de eigenschappen van de verschillende
Digikoppeling profielen weer. Ten behoeve van het CPA register is de kolom 'CPA
Creation' toegevoegd. Voor alle profielen wordt twee-zijdig TLS gebruikt op
transport niveau (HTTPS).

| Profile Names                       | Transport characteristics |              |          |        |           |             |
| ----------------------------------- | ------------------------- | ------------ | -------- | ------ | --------- | ----------- |
| Digikoppeling ebMS2                 | CPA Creation              | 2-zijdig TLS | Reliable | Signed | Encrypted | Attachments |
| Best Effort                         | osb-be                    | √            | N/A      | ―      | ―         | Optional    |
| Reliable Messaging                  | osb-rm                    | √            | √        | ―      | ―         | Optional    |
| Best Effort – Signed<sup>1</sup>    | osb-be-s                  | √            | N/A      | √      | ―         | Optional    |
| Reliable – Signed<sup>1</sup>       | osb-rm-s                  | √            | √        | √      | ―         | Optional    |
| Best Effort – Encrypted<sup>1</sup> | osb-be-e                  | √            | N/A      | √      | √         | Optional    |
| Reliable – Encrypted<sup>1</sup>    | osb-rm-e                  | √            | √        | √      | √         | Optional    |

N/A = Not applicable<br>
<sup>1</sup> End-to-End Security

Met betrekking tot CPA-creatie: zie [[[#deployment-and-processing-requirements-for-cpas]]].

## Berichtuitwisselpatronen

Deze specificatie ondersteunt zowel One Way als Two Way
bericht-uitwisselpatronen (message exchange patterns, terminologie ontleend aan
[[?ebMS3]]). One Way uitwisselingen ondersteunen bedrijfstransacties voor
informatie­verspreiding en notificaties, die geen antwoordbericht
veronderstellen. Two Way uitwisselingen ondersteunen bedrijfstransacties van het
type Vraag-Antwoord, Verzoek-Bevestig, Verzoek-Antwoord en Handelstransacties
(zie [[?UMMR10]], [[?UMMUG]] voor informatie over het concept bedrijfstransactie
patronen). In het geval van tweewegsverkeer leggen de ebXML headervelden
(MessageId, RefToMessagId en ConversationId) de relatie tussen request berichten
en de corresponderende response berichten vast.

Deze specificatie gebruikt uitsluitend een Push binding aan het HTTPS protocol.
Dat wil zeggen dat het retourbericht in een tweewegscommunicatie via een
afzonderlijke HTTPS connectie verloopt, die is geïnitieerd vanuit de verzender
(=de beantwoorder). Het initiële bericht is dan verzonden in een eerdere HTTPS
connectie, die afgesloten is na succesvolle overdracht van het heengaande
bericht.

De keuze van het te gebruiken profiel is onafhankelijk van het uitwisselpatroon.
Het heengaande bericht en (in een tweewegsuitwisseling) het teruggaande bericht
kunnen naar keuze gebruik maken van het Best Effort profiel of het Reliable
Messaging profiel.

## Beveiligingsaspecten

Deze specificatie maakt gebruik een aantal standaarden op het gebied van
beveiliging en voldoet op het moment van schrijven aan geldende richtlijnen en
best practices. Aangezien in de loop der tijd kwetsbaarheden kunnen worden
ontdekt in de cryptografische algoritmen waarop deze standaarden zijn gebaseerd,
is het van belang dat deze specificatie regelmatig op geldigheid hiervan wordt
bezien. De specifieke toegepaste referenties zijn beschreven in 
[[[DK-beveiliging]]].

## Format van dit document

Het OASIS Implementation, Interoperability en Conformance (IIC) Technical
Committee (TC) heeft voor deployment specificaties een sjabloon opgesteld
[[?Deployment Guide 1.1]]. Dat sjabloon is al eerder toegepast door bepaalde
sectoren zoals handel (GS1) en gezondheidszorg (HL7), en wordt daarmee een
standaard manier van het beschrijven van configuraties. Dit document is
opgesteld aan de hand van dat sjabloon. Het is slechts een summiere beschrijving
van het specifieke gebruik van ebXML Messaging en bevat geen
achtergrondinformatie, motivatie, voorbeelden en andere informatie die nuttig is
voor het in de praktijk toepassen van deze specificatie.

Dit document is direct afgeleid van [[?Deployment Guide 1.1]] en om praktische
redenen (grotendeels) in het Engels opgesteld. Leveranciers van producten en
diensten rond ebXML Messaging zijn bekend met dit sjabloon doordat het ook in
andere sectoren wordt gebruikt. Leveranciers kunnen aan de hand van dit sjabloon
eenvoudig nagaan in hoeverre hun product voldoet aan de gestelde eisen.

Dit document is niet (geheel) zelfstandig te lezen maar bedoeld om geraadpleegd
te worden samen met de technische specificatie [EBXML-MSG].

