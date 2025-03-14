# Wat is Digikoppeling?

## Inleiding

Zoals een brief in een envelop gaat voor verzending, zo gaat een elektronisch bericht in een digitale ‘envelop’. Digikoppeling is de standaard digitale ‘envelop’ voor het gestructureerd en gecontroleerd uitwisselen van berichten tussen (semi-)overheidsorganisaties. Met één Digikoppeling-implementatie kunt u berichten uitwisselen met alle overheden en aansluiten op vrijwel alle e-overheidbouwstenen. Denk hierbij aan bijvoorbeeld basisregistraties zoals LV-WOZ, het Omgevingsloket of Digimelding.

Door Digikoppeling kunnen (semi-)overheidsorganisaties eenvoudiger, veiliger, sneller en goedkoper elkaars gegevens gebruiken dan wanneer alle organisaties los van elkaar bilaterale afspraken zouden maken. Het belang en de omvang van gegevensuitwisselingen in de e-overheid neemt alleen maar toe. Digikoppeling is een onmisbare voorwaarde om die uitwisseling efficiënt uit te voeren.

Digikoppeling is één van de Stelselvoorzieningen. Wat de Stelselvoorzieningen zijn wordt in hoofdstuk 6 beschreven.

## Doel van Digikoppeling

Organisaties willen diensten klantgericht, efficiënt, flexibel en rechtmatig aanbieden aan burgers en bedrijven. Daarvoor moeten zij gegevens en documenten op een generieke manier met elkaar kunnen uitwisselen.

Digikoppeling voorziet in een standaard om deze uitwisseling van gegevens en documenten te definiëren. Met deze logistieke standaardisatie bevordert Digikoppeling de interoperabiliteit tussen   
(semi-)overheidsorganisaties. Digikoppeling richt zich op de 'envelop' van het bericht, niet op de inhoud. Daardoor kan iedere organisatie die Digikoppeling gebruikt, de postverzending onafhankelijk van de inhoud inrichten.

Digikoppeling is primair bedoeld voor gegevensuitwisseling tussen systemen van overheidsorganisaties, in het bijzonder de basisregistraties en landelijke of intersectorale gegevensdiensten. Digikoppeling wordt echter breder ingezet in de (semi-)publieke sector. Digikoppeling is beschikbaar voor elke organisatie die veilig en betrouwbaar gegevens wil uitwisselen met andere organisaties in de publieke sector. Tevens is Digikoppeling beschikbaar voor gebruik in de private sector.

## Scope van Digikoppeling

Om digitale berichten uit te wisselen moeten organisaties op drie niveaus afspraken maken:

- Over de inhoud en betekenis van berichten (payload en eventuele bijlagen): de structuur, semantiek, waardebereiken enzovoort.

- Over de logistiek (envelop): transportprotocollen (HTTP), messaging (SOAP), adressering, beveiliging (authenticatie en encryptie) en betrouwbaarheid.

- Over het transport (netwerk): de protocollen van de TCP/IP stack (TCP voor Transport, IP voor Netwerk) en de infrastructuur, bijvoorbeeld Diginetwerk of Internet.

Digikoppeling richt zich op de logistieke laag van de berichtuitwisseling in de publieke sector. Daarbij conformeert Digikoppeling zich aan de Nederlandse Overheid Referentie Architectuur (NORA) en het European Interoperability Framework.

Digikoppeling is geen netwerk, beveiliging, Enterprise Service Bus (ESB) of gegevensstandaard.

### Definitie Digikoppeling

Definitie van Digikoppeling volgens Logius:

*Digikoppeling faciliteert gegevensuitwisselingen tussen (semi-) overheidsorganisaties door standaardisatie van koppelvlakken. Een koppelvlak is een overeengekomen set middelen en afspraken.*

*Een koppelvlakstandaard regelt de opmaak en de veilige, en zo nodig betrouwbare, verzending en ontvangst van een bericht - met header (de envelop), inhoud en eventuele bijlage(n).*

### Gebruik van Digikoppeling door (semi-)overheidsorganisaties

Figuur 1 geeft de plaats van Digikoppeling weer. Digikoppeling legt een verbinding tussen applicaties van de interne organisatie met een (of meer) externe organisaties zodat berichten kunnen worden uitgewisseld. Een organisatie kan een service aanbieden (de “serviceaanbieder” zoals een houder van een basisregistratie), of een service afnemen (de “serviceafnemer” zoals een gemeente, waterschap, rijksdienst of ZBO). Een Enterprise Servicebus of andere routeringsfunctionaliteit verzorgt de berichtuitwisseling tussen de Digikoppeling adapter en lokale applicaties. De berichtuitwisseling met een externe partij kan rechtstreeks of via een intermediair verlopen. De Digikoppeling-adapter vereist een technisch contract voor de berichtuitwisseling in een zogeheten CPA (zie §5.2) of WSDL (zie §5.3) en een PKIoverheids-certificaat in verband met informatiebeveiliging.

Een serviceregister en voorzieningen (de groene blokken) zijn beschikbaar voor het implementatieproces.

![Gebruik van Digikoppeling door (overheids)organisaties: Digikoppeling verbindt serviceafnemers met serviceaanbieders via Digikoppeling adapters](media/gebruik_van_digikoppeling_door_overheidsorganisaties.png "Gebruik van Digikoppeling door (overheids)organisaties")

| **Meer informatie** | **Zie** | **Doelgroep** |
|---|---|---|
| Over NORA in relatie tot Digikoppeling | Digikoppeling\_Architectuur (§2.2) | [A&D]  [OT&B] |
| Link naar NORA | Online: [noraonline.nl](http://www.noraonline.nl/wiki/NORA_online) (link)  | |
| Link naar European Interoperability Framework | Online: [EIF](http://ec.europa.eu/idabc/en/document/2319/5644.html) (link) | |
| Over hulpmiddelen: Serviceregister, compliance-voorzieningen en CPA register | Hulpmiddelen_bij\_implementatie | [PL] [A&D]  [OT&B] |


## De Digikoppeling standaarden

Digikoppeling is gebaseerd op internationale open standaarden van OASIS en W3C, twee wereldwijde standaardisatie-organen voor open standaarden.

De Digikoppeling-standaarden bestaan uit *Koppelvlakstandaarden*. De koppelvlakstandaarden beschrijven de afspraken die nodig zijn om het berichtenverkeer tussen informatiesystemen mogelijk te maken (zoals onderliggende standaarden). Daarnaast zijn er afspraken gemaakt over de Identificatie en Authenticatie van het berichtenverkeer. Onderstaande paragrafen geven een toelichting welke vormen van berichtuitwisseling er zijn en welke koppelvlakstandaard daarbij van toepassing is.

### Koppelvlakstandaarden voor berichtenuitwisseling

Digikoppeling bestaat uit door de overheid vastgestelde koppelvlakstandaarden. Dit zijn logistieke afspraken om berichten juist te adresseren, leesbaar en uitwisselbaar te maken en veilig en betrouwbaar te verzenden.

Digikoppeling beschrijft vier verschillende, maar aanvullende koppelvlakstandaarden: REST API, ebMS2, WUS en Grote Berichten. In de Digikoppeling-documentatie zijn de koppelvlakstandaarden onafhankelijk van specifieke implementaties beschreven. Dat geeft organisaties de vrijheid om ICT-producten met een aansluiting op Digikoppeling te selecteren uit het aanbod van de markt of zelf iets te ontwikkelen.

De keuze voor het gebruik van de REST API, ebMS2 of WUS standaarden hangt onder meer af van de gewenste berichtenuitwisseling (bevragingen en/of meldingen), of er al gebruik wordt gemaakt van deze standaarden en welke standaarden door ketenpartners worden gebruikt.

De vormen van berichtuitwisseling zijn:

- Bevragingen. Vragen waar direct een reactie op wordt verwacht, ofwel synchroon berichtenverkeer. Hierbij is de snelheid van afleveren belangrijk. Als een service niet beschikbaar is, krijgt de verzender een foutmelding en moet hij later of op een andere manier de informatie opvragen.

- Meldingen. Berichten waar meestal niet direct een antwoord valt te geven, ofwel asynchroon berichtenverkeer. De ontvanger krijgt eerst een bevestiging dat zijn zijn bericht ontvangen is. Later volgt het uiteindelijke antwoord.

Zie de schematische weergave in §5.1.

De koppelvlakstandaarden ondersteunen de volgende vormen van berichtuitwisseling:

- REST API koppelvlakstandaard.

- WUS voor bevragingen.

- ebMS2 voor meldingen(transacties).

- Grote berichten voor het uitwisselen van grote bestanden.

Vanwege interoperabiliteit, eenvoud en overzichtelijkheid onderscheidt Digikoppeling per koppelvlakstandaard een aantal standaardprofielen<sup>2</sup>. Elk profiel bestaat uit vooraf gedefinieerde keuzen over kenmerken als synchroniciteit, beveiliging en betrouwbaarheid voor REST API, WUS of ebMS2. Door toepassing van de Digikoppeling profielen worden deze kenmerken correct afgehandeld en kunnen partijen sneller een koppelvlakstandaard implementeren. De profielen worden nader gespecificeerd in de uitgebreide beschrijvingen van de Digikoppeling koppelvlakstandaarden. Een inleiding over deze koppelvlakstandaarden is al in dit document opgenomen in §5.2, §5.3 en §5.4.

<sup>2</sup>: Een specifieke invulling van een van de Digikoppeling koppelvlakstandaarden die een groep functionele eisen invult. Een koppelvlakstandaard kan daardoor meerdere varianten van communicatie bieden. Het betreft functionele eisen op gebied van betrouwbaarheid, veiligheid en performance.

### Identificatie en Authenticatie

Voor de toepassing van Digikoppeling zijn in het kader van informatiebeveiliging afspraken gemaakt over de Identificatie en Authenticatie van partijen en het gebruik van certificaten. In de documenten _Digikoppeling Identificatie en Authenticatie_ en _Digikoppeling Gebruik en Achtergronden Certificaten_ worden de afspraken nader gespecificeerd. Voor gebruik van REST API koppelvlakken is de OAuth standaard van belang. Het Nederlandse profiel daarop is beschreven in _NL GOV Assurance profile for OAuth 2.0_ en _NL GOV Assurance profile for OpenID Connect_.

![Overzicht van de onderdelen van de Digikoppeling Standaard, de standaard is onderverdeeld in normatieve en ondersteunende onderdelen](media/DK_Specificatie_structuur.svg "Opbouw documentatie Digikoppeling")

<details>
    <summary> Tekstalternatief </summary>
<ul>
	<li>Digikoppeling Standaard
		<ul>
			<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/beheer/">DK Beheermodel en releasebeleid</a>* </li>
			<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/actueel/">DK Overzicht Actuele Documentatie en Compliance</a>* </li>
			<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/architectuur">DK Architectuur</a>*
				<ul>
					<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/idauth/">DK Identificatie en Authenticatie</a>*
						<ul>
							<li><i> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/gbachtcert/">Digikoppeling Gebruik en Achtergronden Certificaten</a></i>† </li>
						</ul>
					</li>
					<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/beveilig/">DK Beveiligingsstandaarden en voorschriften</a>* </li>
					<li>Koppelvlakstandaarden
						<ul>
							<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/wus/">DK Koppelvlakstandaard WUS</a>*
								<ul>
									<li><i><a href="https://publicatie.centrumvoorstandaarden.nl/dk/bpwus">Best-practice WUS</a></i>† </li>
								</ul>
							</li>
							<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/ebms/">DK Koppelvlakstandaard ebMS2</a>*
								<ul>
									<li> <i><a href="https://publicatie.centrumvoorstandaarden.nl/dk/bpebms">Best-practice ebMS2</a></i>† </li>
								</ul>
							</li>
							<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/restapi/">DK Koppelvlakstandaard REST API</a>*
							</li>
							<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/gb/">DK Koppelvlakstandaard Grote Berichten</a>*
								<ul>
									<li> <i><a href="https://publicatie.centrumvoorstandaarden.nl/dk/bpgb">Best-practice Grote Berichten</a></i>†</li>
								</ul>
							</li>
						</ul>
					</li>
				</ul>
			</li>
			<li>
    <i><a href="https://publicatie.centrumvoorstandaarden.nl/dk/watisdk/">Wat is Digikoppeling</a></i>†
  </li>
		</ul>
	</li>
</ul>
<p>* Normatief document</p>
<p>† Ondersteunend document</p>
</details>


De Digikoppeling standaarden (zie Figuur 2) zijn nader uitwerkt in verschillende documenten. In Tabel 2 staat een overzicht om welke documenten het gaat en is kort toegelicht wat deze beschrijven.

| **Document** | **Wat beschrijft deze?** | **Doelgroep** |
|---|---|---|
| Digikoppeling Koppelvlakstandaard WUS | Het gebruik van WUS voor bevragingen en de WUS profielen. | [A&D]  [OT&B] |
| Digikoppeling Koppelvlakstandaard ebMS2 | Het gebruik van ebMS2 voor meldingen en de ebMS2 profielen |[A&D]  [OT&B] |
| Digikoppeling Koppelvlakstandaard Grote Berichten | Voor de uitwisseling van grote berichten maakt gebruik van WUS met HTTPS bestandsoverdracht of ebMS2 met HTTPS bestandsoverdracht |[A&D]  [OT&B] |
| Digikoppeling Koppelvlakstandaard REST API | Het gebruik van REST API koppelvlakken |[A&D]  [OT&B] |
| Identificatie en Authenticatie en Gebruik en Achtergrond Digikoppeling Certificaten  | Beschrijft de identificatie van partijen, het opzetten van een tweezijdige beveiligde TLS-verbinding en het ondertekenen en versleutelen van berichten en bijlagen.  | [A&D]  [OT&B] |

Tabel 2: Documenten met inhoudelijke uitwerking van de Digikoppeling-standaarden

## Succesvolle toepassingen met Digikoppeling

Op de Logius website zijn voorbeelden geplaatst van succesvolle toepassingen met Digikoppeling. Zie Online: [Logius](https://www.logius.nl/diensten/digikoppeling) (link).
