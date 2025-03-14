# OIN en PKIO in relatie tot OAuth NLGov profiel

> Concept beschrijving dd 22-08-2024 - door Martin van der Plas

## Inleiding

De nieuwe versie van het OAuth NL profiel voegt onder meer het Client credentials profiel toe. 
Zie ook v1.1.0-rc.2 op :  https://logius-standaarden.github.io/OAuth-NL-profiel/ 

Op basis van deze toevoeging kwamen, na vaststelling door de werkgroep en de publieke consultatie, aanvullende vragen over het gebruik van PKIO certificaten en het OIN vanuit het onderwijs domein. De Edukoppeling werkgroep gebruikt voor het definiëren van een aantal OAuth best practices[1] ook het NL GOV Assurance profile for OAuth 2.0 (NL GOV). 

Aangezien het NLGov OAuth profiel is opgesteld op basis van het iGov profiel is het soms wat lastig om te doorgronden wat de precieze vereisten zijn. Ook is de materie inzake oauth relatief complex en wordt er nogal eens wat impliciete voorkennis verondersteld. Dit atikel dient dan ook niet als vervanging van de standaard of als aanvulling daarop maar meer als richtlijn of toelichting. 

## Context:

**OIN refenties**
De standaard heeft de volgende relevante verwijzingen naar het OIN die hieronder verder worden toegelicht
- Par 2.3.4 https://logius-standaarden.github.io/OAuth-NL-profiel/#client-keys beschrijft de connectie van de client met de Authorization server en vereist simpel gezegd dat clients een keypair moeten hebben om zich te authenticeren bij het token endpoint. in onze additionele content eisen we dat hiervoor een PKIO certificaat met OIN moet worden gebruikt als de authorization server, resource server en client niet allemaal van dezelfde organisatie zijn. Aanvullend eisen we dat clients hun public key moeten registreren in de client metadata
- Par 3.2 https://logius-standaarden.github.io/OAuth-NL-profiel/#connections-with-protected-resources beschrijft hoe resources connecten met de authorization server. het vereist dat de authorization server JWT tokens uitgeeft. Ook hier eisen we dat hiervoor een PKIO certificaat met OIN moet worden gebruikt als de authorization server, resource server en client niet allemaal van dezelfde organisatie zijn. 

**PKI referenties**
De standaard heeft de volgende relevante verwijzingen naar PKI die hieronder verder worden toegelicht:
- 

## Intenties:

- Clients in de client credentials flow zijn pre-registerd [Step 1. Client Authentication](https://logius-standaarden.github.io/OAuth-NL-profiel/#step-1-client-authentication)
- Private_key_jwt is naast tls_client_auth toegestaan. In beide situaties is een Private key nodig om ofwel de JWT te signen ofwel de mTLS connectie op te zetten. Indien dus de PKIO verplicht is is in beide gevallen een OIN inzichtelijk bij de authorization server.   
- verbindingen tussen client, resource en authorization server zijn altijd versleuteld met TLS en indien PKIO verplicht is dus ook altijd the identificeren (op TLS niveau met de OIN uit het subjectSerialNumber veld van de public key)
- 

## Conclusie:

- een kwaadwillende kan nooit met een self signed certificaat een client faken die met een ander certificaat is geregistreerd bij de authorization server

## Achtergrondinformatie:

### Vragen DUO:

De stapjes binnen Edustandaard over het toepassen van OAUTH gaan langzamer dan ik zou willen, maar zeker wel de goede kant op. Ondertussen heb ik ook een nieuwe 'proposed' versie van NL-GOV (10 juli 2024) gezien.  (Terzijde,  er is ook een versie van 13 mei  vindbaar, zonder dat duidelijk is dat die is vervangen, dat kan verwarrend zijn).  Ook in deze nieuwe versie zit iets of juist niet, wat DUO denkt nodig hebben. Vandaar dat ik er een vraag over stel.

Eén van de requirements is redelijk scherp,  "de identiteit (OIN) van de client wordt onomstotelijk vastgesteld".

In deze versie lees ik dit:

2.3.3 In addition to private_key_jwt, the client authentication method tls_client_auth [rfc8705https://logius-standaarden.github.io/OAuth-NL-profiel/#bib-rfc8705] MAY also be used. 2.3.4 Clients using the authorization code grant type or direct access clients using the client credentials grant type MUST have a public and private key pair for use in authentication to the token endpoint. These clients MUST register their public keys in their client registration metadata by either sending the public key directly in the jwks field or by registering a jwks_uri that MUST be reachable by the authorization server 2.3.4 In case the Authorization Server, Resource Server and client are not operated under responsibility of the same organisation, each party MUST use PKIoverheid certificates with OIN.

Hieruit spreekt  dezelfde intentie uit als onze requirement. Dat is goed. Maar wordt het waar gemaakt?  Het idee van DUO:

- Bij mTLS wel. - De server leest de metadata van het PKI certificaat uit. Haalt daaruit het root - certificaat (NL overheid) en ook het OIN (het SSN-veld) van de certificaat-houder.

- Bij private_key_jwt niet. - De publieke sleutel wordt uitgelezen met het  jwks- of jwks_uri-veld. Deze versie van NL-GOV beschrijft een JSON Web Key Set (JWK set) van metagegevens dat daarmee toegankelijk is.  Dus wel de publieke sleutel, maar niet het root-certificaat en  het SSN-veld met OIN.  Een kwaadwillende kan met een self signed certificaat een legitieme client faken.

Als dat inderdaad zo is, dan is private-key-jwt ongeschikt voor client credentials en hoort het niet thuis in dit deel van deze standaard.

---



###  Interpretatie Kennisnet:

Mijn interpretatie bij de toepassing van een aantal NL GOV aspecten is als volgt:

**Toepassing private_key_jwt voor client authenticatie en PKIo certificaat voor ondertekening:**

1. De ondertekening is de jwt. De bij registratie opgenomen client_id en public key van client maken het mogelijk voor AS om ondertekening te verifiëren[2].      
- Hiermee wordt bewezen dat client over private key PKIo beschikt (proof-of-possession authenticatie). Ik ga er hierbij vanuit dat client authenticatie van gelijkwaardig niveau is als mTLS. Tevens kan AS client_id aan OIN relateren (identificatie). 


**Toepassing PKIo (client en AS/RS niet onder beheer van dezelfde partij)**

1. Omdat client niet door dezelfde partij wordt beheerd als AS/RS schrijven we conform NL GOV OAuth voor dat PKIo gebruikt moet worden.         
- In combinatie met toepassing van private_key_jwt client authenticatie (en dus geen mTLS) betekent dit dus dat ondertekening van privat_key_jwt met PKIo certificaat wordt toegepast.   2.  Het PKIo certificaat authenticeerd de rechtspersoon. Met het OIN in certificaat kan de rechtspersoon geïdentificeerd worden.  Met het door de AS uitgegeven client_id kan een client (als onderdeel van het applicatielandschap van rechtspersoon) geïdentificeerd worden.
- Er is dus identificatie op 2 niveaus, OIN en client_id. NL GOV OAuth vereist niet dat de client_id het OIN is. Dit wordt echter niet expliciet aangegeven. Binnen het onderwijs biedt het voordelen op verschillende niveaus te kunnen identificeren, maar toepasbaarheid is onduidelijk. Ook onduidelijk of op dit niveau niet uniformiteit/standaardisatie wenselijk is.   
3.  Voor AS server geldt bij toepassing van private_key_jwt dus GEEN mTLS maar wel toepassing van PKIo voor serverbeveiliging (SSL/TLS) en bepaalde TLS versie.         
- In NL GOV staat bij paragraaf 2.4.1 Requests to the Protected Resource: Authorized requests MUST be made over TLS, and clients MUST validate the protected resource server's certificate. Ik neem aan dat dit ook voor requests naar AS geldt als client authenticatie obv private_key_jwt wordt toegepast.
- In deze situatie wordt AS en RS serverbeveiliging vereist op basis van TLS en PKIo. Er gelden dan ook aanvullende eisen rond TLS versie en ciphert suites (NCSC/DK) die expliciet aangegeven moeten worden.

**Discovery**

1. NL GOV OAuth - Discovery (paragraaf 3.1.5) stelt :"The authorization server MUST provide an [OpenID Connect service discovery] [OpenID.Discovery] endpoint listing the components relevant to the OAuth protocol".  \      
- Er is onduidelijkheid wat de exacte scope / betekenis is van discovery. In het NL GOV OAuth profiel lijkt dit expliciet betrekking te hebben op de AS meta data. Het is onduidelijk of discovery ook betrekking heeft op meta data van client (het toepassen van een uri in plaats van opname in een jwks veld). Bij client authenticatie op basis van private_key_jwt kan zowel een jwks_uri gebruikt worden als een jwks veld. Is met het toepassen van discovery de toepassing van een jwks_uri logisch en de toepassing van private_key_jwt het jwks veld[3] onlogisch (en visa versa indien discovery juist niet vereist wordt zoals bij Edukoppeling OAuth Best Practices)?



---

### Links: 



[1] https://www.edustandaard.nl/app/uploads/2024/07/2024-07-16-Edukoppeling-OAuth-Best-Practices-v1.0.pdf

[2] Een client moet zijn publieke sleutel vooraf registreren bij een autorisatieserver, zodat de server hiermee de ondertekening kan verifiëren.

[3] En hierin ook het PKIoverheid cert als x5c parameter (X. 509 Certificate Chain) en niet x5u. Dit bericht kan informatie bevatten die niet voor u is bestemd. Indien u niet de geadresseerde bent of dit bericht abusievelijk aan u is toegezonden, wordt u verzocht dat aan de afzender te melden en het bericht te verwijderen. De Staat aanvaardt geen aansprakelijkheid voor schade, van welke aard ook, die verband houdt met risico's verbonden aan het elektronisch verzenden van berichten. This message may contain information that is not intended for you. If you are not the addressee or if this message was sent to you by mistake, you are requested to inform the sender and delete the message. The State accepts no liability for damage of any kind resulting from the risks inherent in the electronic transmission of messages.