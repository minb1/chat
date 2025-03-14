# Inleiding

## Doel en doelgroep

Alle Digikoppeling profielen die op ebMS2 gebaseerd zijn, moeten zich conformeren aan de Koppelvlakstandaard ebMS2. Dit document is een aanvulling hierop. Het heeft als doel ontwikkelaars te adviseren en te informeren over de huidige werkwijze bij het toepassen van Digikoppeling Koppelvlakstandaard ebMS2 – deze informatie geldt dus alleen voor de ebMS2-variant.

Het document is bestemd voor Architecten, integratie specialisten en ontwikkelaars van webservices die zijn aangesloten op Digikoppeling. Het gaat hierbij om zowel (service) aanbieders als (service) afnemers. Zie onderstaande tabel bij welke taken dit document ondersteunt.

| Afkorting | Rol                             | Taak                                                                                                       | Doelgroep? |
|-----------|---------------------------------|------------------------------------------------------------------------------------------------------------|------------|
| [MT]      | Management                      | Bevoegdheid om namens organisatie (strategische) besluiten te nemen.                                       | **Nee**    |
| [PL]      | Projectleiding                  | Verzorgen van de aansturing van projecten.                                                                 | **Nee**    |
| [A&D]     | Analyseren & ontwerpen (design) | Analyseren en ontwerpen van oplossings-richtingen. Het verbinden van Business aan de IT.                   | **Nee**    |
| [OT&B]    | Ontwikkelen, testen en beheer   | Ontwikkelt, bouwt en configureert de techniek conform specificaties. Zorgen voor beheer na ingebruikname.  | **Ja**     |

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


Alle met de kleur groen aangegeven documenten vallen onder het beheer zoals geformaliseerd in het Beheermodel en releasebeleid.

## Doel en scope van Digikoppeling

Voor de Overheid als geheel is interoperabiliteit tussen een groot aantal serviceaanbieders en serviceafnemers van essentieel belang. Die grootschalige interoperabiliteit wordt bereikt door sterke standaardisatie van het koppelvlak tussen de communicatiepartners.

Deze communicatie vindt plaats in het domein van Digikoppeling, en daarbij worden Digikoppeling Koppelvlakstandaarden toegepast. Dat is een beperkte set van standaarden waaruit onder gedefinieerde omstandigheden gekozen kan worden.

Digikoppeling biedt de mogelijkheid om op deze gestandaardiseerde wijze berichten uit te wisselen tussen service aanbieders en service afnemers. Digikoppeling richt zich voornamelijk op uitwisselingen tussen overheidsorganisaties maar er zijn ook implementaties bekend van de toepassing van Digikoppeling bij uitwisseling tussen bedrijven en overheden.

### Uitwisseling binnen Digikoppeling

De uitwisseling tussen partijen is in drie lagen opgedeeld:

1. Inhoud (gegevens): deze laag bevat afspraken over de inhoud van het uit te wisselen bericht, dus de structuur, semantiek en waardebereiken <BR>
Digikoppeling houdt zich niet met de inhoud bezig, '_heeft geen boodschap aan de boodschap_'.

2. Logistiek (processen): op deze laag bevinden zich de afspraken betreffende transportprotocollen (HTTP/TLS), messaging (SOAP), adressering, beveiliging (authenticatie en encryptie) en betrouwbaarheid. Dit is laag van Digikoppeling.

3. Transport (techniek): deze laag verzorgt het daadwerkelijke transport van het bericht (TCP/IP).

> Digikoppeling richt zich uitsluitend op de logistieke laag.

## Opbouw van dit document

[Hoofdstuk 1 bevat een aantal algemene inleidende onderwerpen.](#inleiding)

[Hoofdstuk 2 bevat de aanbevelingen, werkwijze en best practices.](#werkwijze-aanbevelingen-best-practices)

[Hoofdstuk 3 gaat in op de kenmerken van een CPA.](#cpa-gebruik-en-kenmerken)

[Hoofdstuk 4 gaat over bericht volgordelijkheid.](#het-gebruik-van-berichtvolgordelijkheid)

> Begrippen en afkortingen worden toegelicht in het document [[Digikoppeling-Architectuur]].  
> Dit document is onderdeel van de Digikoppeling standaard.

<div class="note">Dit document en alle andere Digikoppeling documentatie is beschikbaar op [de website van Logius](http://www.logius.nl/digikoppeling).
</div>
