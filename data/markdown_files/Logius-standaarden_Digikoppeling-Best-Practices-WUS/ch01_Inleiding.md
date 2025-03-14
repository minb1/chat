# Inleiding

## Doel en doelgroep

Alle Digikoppeling webservices die op WUS gebaseerd zijn, moeten conformeren aan Digikoppeling Koppelvlakstandaard WUS. Dit document is een aanvulling hierop. Het heeft als doel ontwikkelaars te adviseren en te informeren over de huidige werkwijze bij het toepassen van Digikoppeling Koppelvlakstandaard WUS – deze informatie geldt dus alleen voor de WUS-variant.

Het document is bestemd voor ontwikkelaars van webservices, die Digikoppeling toepassen. Het gaat hierbij om zowel service providers als service requesters (clients).

## Opbouw Digikoppeling documentatie

Digikoppeling is beschreven in een set van documenten. Deze set is als volgt opgebouwd:

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


## Digikoppeling

### Doel en scope van Digikoppeling

Voor de Overheid als geheel is interoperabiliteit tussen een groot aantal serviceaanbieders en serviceafnemers van essentieel belang. Die grootschalige interoperabiliteit wordt bereikt door een sterke standaardisatie van het koppelvlak tussen de communicatiepartners.

Deze communicatie vindt plaats in het domein van Digikoppeling, en daarbij worden Digikoppeling koppelvlakstandaarden toegepast. Dat is een zeer beperkte set van standaarden waaruit onder gedefinieerde omstandigheden gekozen kan worden.

Digikoppeling biedt de mogelijkheid om op die sterk gestandaardiseerde wijze berichten uit te wisselen tussen serviceaanbieders (service providers) en serviceafnemers (service requesters of clients). Digikoppeling richt zich (in elk geval voorlopig) uitsluitend op uitwisselingen tussen overheidsorganisaties.

### Uitwisseling binnen Digikoppeling

De uitwisseling tussen service providers en - requesters is in drie lagen opgedeeld:

- Inhoud: deze laag bevat afspraken over de inhoud van het uit te wisselen bericht, dus de structuur, semantiek en waardebereiken . Digikoppeling houdt zich niet met de inhoud bezig, “heeft geen boodschap aan de boodschap”.

- Logistiek: op deze laag bevinden zich de afspraken betreffende transportprotocollen (HTTP), messaging (SOAP), adressering, beveiliging (authenticatie en encryptie) en betrouwbaarheid. Dit is de laag van Digikoppeling.

- Transport: deze laag verzorgt het daadwerkelijke transport van het bericht.

Digikoppeling richt zich uitsluitend op de logistieke laag. Deze afspraken komen in de koppelvlakstandaarden en voorzieningen.

## Opbouw van dit document

Hoofdstuk 1 bevat een aantal algemene inleidende onderwerpen.

Hoofdstuk 2 bevat aanbevelingen, werkwijzes en Best Practices.
