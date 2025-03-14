# Inleiding

## Doel en Doelgroep

Dit document beschrijft de functionele specificaties voor de Digikoppeling
koppelvlak standaard WUS

Het document is bestemd voor architecten en ontwikkelaars die op basis van WUS
gegevens willen uitwisselen via Digikoppeling.

Alle Digikoppeling webservices die op WUS gebaseerd zijn, moeten conformeren aan
de koppelvlakstandaard WUS. Deze wordt tot in detail in dit document
gespecificeerd. Doel van dit document is ontwikkelaars te informeren wat deze
koppelvlakstandaard nu precies inhoudt en waar zij zich aan moeten conformeren.
Het document is bestemd voor architecten en ontwikkelaars die op basis van WUS
gegevens willen uitwisselen via Digikoppeling. Het gaat hierbij om zowel service
providers als service requesters (clients).

## Opbouw Digikoppeling documentatie

Digikoppeling is beschreven in een set van documenten. Deze set is als volgt
opgebouwd:

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
							<li> <a href="https://publicatie.centrumvoorstandaarden.nl/dk/restapi/">DK Koppelvlakstandaard REST-API</a>*
								<ul>
									<li><i>Best-practice REST-API</i>† </li>
								</ul>
							</li>
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


Dit document beschrijft de WUS-variant van de koppelvlakstandaarden. Naast de
Koppelvlakstandaard WUS zijn er ook de ebMS2- en REST-API-standaarden. Deze worden in een aparte
documenten beschreven.

## Koppelvlak & koppelvlakstandaard

Een koppelvlak is een interface die volgens standaarden de gegevensuitwisseling
vastlegt. Het werken met vaste standaarden is essentieel voor een koppelvlak.
Hierdoor wordt implementatie vergemakkelijkt. Ook wordt het mogelijk diverse
soorten berichten door te sturen met een grote mate van interoperabiliteit,
omdat via de standaard afspraken over hun inhoud gemaakt is.

Eén van de belangrijkste eisen die door de overheid gesteld worden bij de
inrichting van generieke voorzieningen, is dat er niet veel maatwerk ontwikkeld
hoeft te worden, maar dat er van “off the shelf” commercieel of Open source
geleverde software gebruik gemaakt kan worden. Voor Digikoppeling, dus voor de
logistieke laag, betreft dat het niet willen ontwikkelen van software voor de
adapters.

Dit doel kan bereikt (benaderd) worden doordat gekozen wordt voor internationale
(de jure of de facto) vastgelegde standaarden, die door “alle” leveranciers
interoperabel zijn geïmplementeerd.

Een andere eis is dat met name afnemers gebruik kunnen maken van één “stekker”
(één logistiek koppelpunt). Aanbieders dienen hiervoor de nodige voorzieningen
te treffen.

Een koppelvlakspecificatie beschrijft de eisen die gesteld worden aan de
adapters om interoperabel met elkaar te kunnen communiceren. Digikoppeling gaat
over logistiek, dus over de envelop en niet over de inhoud. De hele set
informatie die tezamen nodig is voor een complete generieke Digikoppeling
koppelvlakdefinitie bestaat uit:

- Interfacedefinitie “on the wire”, (voorbeeld)listing van SOAP headers en
    informatie over velden en hun specifieke inhoud.

De voor Digikoppeling vereiste interoperabiliteit van de WUS standaarden van
OASIS en W3C wordt gebaseerd op de profielen (en tests) van OASIS WS-BRSP
(voorheen WS-I).

## Opbouw van dit document

Hoofdstuk 1 bevat een aantal algemene inleidende onderwerpen.

Hoofdstuk 2 bevat de kern van de standaard. Deze is onderverdeeld naar
onderwerpen/gebieden: WSDL, WS-Addressing, naamgeving, beveiliging,
betrouwbaarheid en binaire data. Het identificeert de gekozen internationale
(WS-I) profielen die dienen als fundament voor de Digikoppeling
Koppelvlakstandaard WUS. Die keuzes, de nadere invullingen voor Digikoppeling
binnen de ruimte van die standaarden/profielen en specifieke aandachtspunten bij
de keuzes, vormen tezamen de “voorschriften” per onderwerp.

Hoofdstuk 3 definieert de resulterende Digikoppeling WUS profielen.

Gehanteerde terminologie: Glossary

Voor de definities die binnen het Digikoppeling project gehanteerd worden, zie
de ‘Digikoppeling Glossary’ via de onderstaande website.

Website

Dit document en andere documentatie is beschikbaar op
[www.logius.nl/digikoppeling](http://www.logius.nl/digikoppeling).

