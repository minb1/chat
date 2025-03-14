# Werkwijze/Aanbevelingen/Best Practices

## EB001 ebMS2 Producten

Gebruik een ebMS2 product met aantoonbare ervaring binnen het Digikoppeling domein. Op de [website van Logius is een overzicht van Digikoppeling leveranciers](https://logius.nl/domeinen/gegevensuitwisseling/digikoppeling/ict-leveranciers) te vinden. De [GEMMA softwarecatalogus](https://www.softwarecatalogus.nl/pakketten?zoek=digikoppeling+ebms) van de VNG geeft een overzicht van softwareproducten voor de gemeentelijke markt die de Digikoppeling ebMS koppelvlakspecificatie ondersteunen.

## EB002 CPA Gebruik

Voor het definiëren van een CPA geldt het volgende advies:

1. Raadpleeg het hoofdstuk 3 'CPA Gebruik en Kenmerken'.

1. Maak gebruik van het online CPA Register<sup>1</sup>, zie het [CPA Register gebruikershandleiding](https://www.logius.nl/domeinen/gegevensuitwisseling/digikoppeling/documentatie/cpa-register-gebruikershandleiding)

> <sup>1</sup>. Het online CPA Register wordt gehost door Justid en is beschikbaar op [https://cparegister.minvenj.nl/logius](https://cparegister.minvenj.nl/logius)

## EB003 Productie- en ontwikkelomgevingen

Het is raadzaam om test- en productieservices ('OTAP') op aparte machines onder te brengen om het onderscheid tussen beide helder te houden. Geef ze een eigen DNS naam (en dus verschillende PKI overheid certificaten), bijvoorbeeld door het gebruik van verschillende subdomeinnamen.

## EB004 PartyId postfix

Voorzie de PartyId van een postfix voor het onderscheid tussen test- en productieservices ('OTAP'). De naamgevingsconventie is hierbij:

- Ontwikkelomgeving met de postfix ‘_O’

- Testomgeving met de postfix ‘_T’

- Acceptatieomgeving met de postfix ‘_A’

- Productieomgeving zonder postfix (het oorspronkelijke nummer).

Samenstellingen zijn ook mogelijk, bijvoorbeeld de postfix ‘_OTA’ als er één specifiek adres gebruikt wordt voor de ontwikkel-, test-, en acceptatieomgeving. Aangezien Digikoppeling een strikte scheiding tussen test en productie nastreeft zou een combinatie van productie met andere omgevingen nooit moeten voorkomen<sup>2</sup>.

> <sup>2</sup>. De scheiding komt ook. tot uitdrukking in het gebruik van een andere certificaat-root voor productie en andere omgevingen. Zie hiervoor het document “Gebruik en achtergrond Digikoppeling-certificaten”.

## EB005 Certificaten

Vanuit het oogpunt van beveiliging is het dringende advies om aparte certificaten te gebruiken voor de 'OTA' omgevingen aan de ene kant en de productie ('P') omgeving aan de andere kant.

Ook wordt geadviseerd om autorisaties te baseren op het OIN dat uit het certificaat verkregen wordt. Indien autorisaties plaatsvinden met het PartyId (dat ook het OIN bevat) dient zeker gesteld te worden dat de beide voorkomens van het OIN (certificaat en PartyId) identiek zijn of dat de organisatie in het certificaat mag handelen namens de organisatie in het PartyId. Dit kan met geautomatiseerde controle (bijvoorbeeld bij ontvangst/verzending van een bericht op de TLS-offloader) of door handmatige controle (bijvoorbeeld bij het aanmaken van het CPA dat in de Digikoppeling-adapter ingelezen wordt).

## EB006 Service & Action naamgeving

Advies 1: gebruik een functionele naam voor de naamgeving van de Service. Verwerk de versie in de service naam. De servicenaam mag de URN zijn.

<aside class="example">
Voor Digimelding  wordt een service gedefinieerd voor de verwerking van de berichten tussen de afnemer en Digimelding en tussen de registratiehouders en de Digimelding. De service krijgt de naam:

Digimelding:1:0
</aside>

Advies 2: als er gebruik gemaakt wordt van het CPA Register, gebruik dan als Identificerende naam de naam van de service (het laatste onderdeel van het pad in de URN.)

<aside class="example">
Voor Digimelding wordt een service gedefinieerd voor de verwerking van de berichten tussen de afnemer en Digimelding en tussen de registratiehouders en Digimelding. De service wordt opgeslagen in het CPA Register met de Identificerende naam:

Digimelding.1.0
</aside>

Met deze Identificerende naam kan een afnemer een CPA laten maken op basis van de naam (zie 'Digikoppeling CPA Creatievoorziening Handleiding' op Digikoppeling website).

Advies 3: gebruik een functionele naam (liefst een werkwoord) voor de naamgeving van de Actions. Denk eraan dat een Service meerdere Actions mag bevatten (meldingen).

<aside class="example">
Voor de Digimelding wordt in een service de Action gedefinieerd voor het terugmelden van een geconstateerde foutieve registratie. De naam voor de service is:
terugmelden
</aside>

In Digikoppeling Koppelvlakstandaard WUS staat bij voorschrift WW003 hoe de payload in de SOAP body opgenomen moet worden: op basis van de 'document-literal style'. Bij document-literal mag de payload slechts 1 XML element bevatten; hierbinnen kunnen wel meerdere elementen opgenomen worden. Het is ook bijvoorbeeld mogelijk om meerdere elementen van het type {`http://www.w3.org/2001/XMLSchema`} base64Binary op te nemen binnen dit eerste element. Daarmee ondersteunt deze koppelvlakstandaard het versturen van attachments met binaire data impliciet.

Het wordt sterk aangeraden om voor ebMS2 deze werkwijze over te nemen. De naam van het payload element zal dan gebruikt worden als naam voor de Action.

## EB007 Rollen

Als een overheidsorganisatie in een bepaalde service zowel berichten kan versturen als wel berichten kan ontvangen, ga dan na wat de functionele rol is. In welke hoedanigheid wordt de functie uitgevoerd? Deze functionele rol zal een bepaalde naam hebben. Gebruik dan die naam voor de rol in de CPA.

<aside class="example">
Een abonnementen service biedt de mogelijkheid om organisaties zich te laten inschrijven op een topic, of om zich te laten uitschrijven op een topic. De abonnementen service zal op gezette tijden een nieuw item van een topic naar een afnemer sturen. Merk op dat de berichten in alle gevallen meldingen zijn.

Vanuit een eenvoudig oogpunt zou je kunnen zeggen dat de organisatie die de abonnementen service implementeert, zowel berichten verstuurt als ontvangt:

- ontvangt, voor het verwerken van de aanvragen van de afnemer, en

- verstuurt, voor het verzenden van nieuwe topics.

Vanuit de optiek dat de organisatie een samenhangende verzameling van berichten gedefinieerd heeft voor de implementatie de abonnementen service, is het zinvol om de organisatie die de service aanbiedt een en dezelfde rol te geven: bijvoorbeeld 'TopicHouder'.

De service krijgt de naam “AbonnementenService”. Afnemers krijgen de rol 'Abonnee'. De organisatie die de service implementeert krijgt de rol 'TopicHouder'. De volgende meldingen zijn mogelijk:

- Van 'Abonnee' rol naar 'TopicHouder' rol: melding InschrijvenOpTopic(topic)

- Van 'Abonnee' rol naar 'TopicHouder' rol: melding UitschrijvenOpTopic(topic)

- Van 'Abonnee' rol naar 'TopicHouder' rol: bevraging RaadpleegTopics()

- Van 'TopicHouder' rol naar 'Abonnee' rol: melding NieuwTopicItem()

(Voor de volledigheid: het opvragen van de beschikbare topics is een 'bevraging' op Digikoppeling en zal met WUS gedaan moeten worden.)

De Abonnee kan dus berichten versturen (om zich in- of uit te schrijven), maar ook ontvangen (een nieuw item van een topic). De topic houder kan berichten ontvangen (de in of uitschrijvingen van afnemers), maar ook berichten versturen (de nieuwe items van een topic).

</aside>

## EB008 Overdrachtskarakteristieken

De karakteristieken voor de betrouwbare overdracht worden uitgedrukt in ‘RetryInterval’ en ‘RetryCount’. Digikoppeling Koppelvlakstandaard ebMS2 heeft hiervoor een aanname gemaakt. Evalueer met de betrokken partijen of deze waardes van toepassing zijn. Wijzig de waardes zo nodig. Houd hierbij rekening met zowel de bedrijfsmatige verwerking van de meldingen als met de netwerk karakteristieken (beschikbare bandbreedte, omvang van de payload, quality of service en dergelijke) tussen de twee overheidsinstanties. Raadpleeg voor meer informatie hoofdstuk 3 'CPA Gebruik en Kenmerken'.

## EB009 Vaststelling CPAId

Geef zowel een CPAId als een start- en einddatum op bij het maken van een overeenkomst tussen een service requester en een service provider. Deze invulling moet in overleg met de twee partijen bepaald worden. Raadpleeg voor meer informatie hoofdstuk 3 'CPA Gebruik en kenmerken'.

## EB010 Geldigheidsperiode van een CPA

Laat de start- en einddatum van de CPA mede afhangen van de geldigheid van de gebruikte client- & servercertificaten.

Een CPA voor testdoeleinden kan een korte geldigheidsperiode krijgen, afgestemd op de test periode.

Een geldigheidsperiode langer dan 10 jaar is niet gebruikelijk.

## EB011 MessageOrder en ConversationId

Als de MessageOrder functionaliteit gebruikt wordt, moeten alle betreffende (samenhangende) berichten dezelfde ConversationId krijgen.

Er zijn meerdere, onafhankelijke, berichtstromen mogelijk waar MessageOrder op van toepassing is als elke berichtstroom zijn eigen ConversationId krijgt.

## EB012 MessageOrder en ReliableMessaging

Als MessageOrder gebruikt wordt is dit van invloed op de overdrachtskarakteristieken (EB008). Als een enkel bericht in de overdracht faalt, heeft dit tot gevolg dat alle opvolgende berichten niet verzonden kunnen worden, of afgeleverd kunnen worden. De waardes voor de RetryCount en RetryInterval zullen daarom met zorg gekozen moeten worden.

## EB013 MessageId

De Koppelvlakstandaard ebMS2 schrijft het gebruik van een MessageId voor conform [[RFC2822]], in de vorm van “[UUID@URI](mailto:UUID@URI)”.

De URI in de MessageId kan ook het domein kan zijn van Digikoppeling messagehandler.

## EB014 Meerdere PartyId's

Als een grote organisatie, bestaande uit meerdere deel-organisaties, via een gateway (in de algemene zin, niet te verwarren met Digikoppeling Gateway) aangesloten wordt op Digikoppeling, kan de situatie ontstaan dat de deel-organisaties een ander OIN hebben dan het OIN van de gehele organisatie. Ook kan de situatie ontstaan dat voor de cliënt authenticatie van de gateway er maar 1 certificaat gebruikt kan worden (namelijk die van de organisatie als geheel).

Het is toegestaan om op transport niveau het cliënt certificaat te gebruiken voor de authenticatie van de organisatie als geheel. Als waarde voor de PartyId mogen OIN's gebruikt worden die afwijken van het OIN in het authenticatie certificaat. Het is aan de samenwerkende partijen om er op toe te zien dat de juiste PartyId's gebruikt worden. In dergelijke situaties is het tevens toegestaan om PartyId's te gebruiken die afwijken van de gangbare OIN's, mits er een ander PartyId type aangegeven wordt. (Voor de OIN geldt urn:osb:oin.) Vanuit Digikoppeling wordt in dergelijke gevallen een dringend advies gegeven om de deel-organisaties een eigen OIN te laten aanvragen. (Het toestaan van een eigen PartyId waarde in de communicatie op Digikoppeling tussen de Nederlandse overheden wordt gezien als een tijdelijke situatie).

## EB015 SyncReplyMode

Het Digikoppeling ebMS2 Reliable Messaging (RM) profiel vereist dat berichten bevestigd worden met een *acknowledgement* bericht. Binnen het Digikoppeling ebMS2 RM-profiel moet deze acknowledgement *asynchroon* worden verzonden. Een Digikoppeling oplossing houdt hiervoor een administratie bij zodat de acknowledgement aan het initiële *MessageId* van het bericht kan worden kan worden gerefereerd

Bij het gebruik van het uitwisselen van zeer hoge volumes van berichten in beperkte tijd via het Digikoppeling ebMS RM-profiel *kan* de overhead van het asynchroon bevestigen van het bericht via asynchrone acknowledgement (te) groot worden. Vandaar dat sinds versie 3.3 van dit profiel het gebruik van SyncReply Profile in de Digikoppeling ebMS2 specificatie wordt toegestaan voor een aantal uitzonderingsgevallen. Vanaf deze versie is het in bepaalde gevallen mogelijk om een bericht synchroon – dus in dezelfde http-sessie- te bevestigen met een acknowledgement (of een foutsituatie synchroon te beantwoorden met een *errormessage*)

**Toepassen van Synchrone acknowledgement is uitzondering**

Het gebruik van Syncreply wordt in de regel afgeraden. Tot aan versie 3.3 van Digikoppeling ebMS was een invulling van het attribuut SyncReplyMode in het CPA anders dan “none” niet toegestaan, met als onderbouwing: Asynchronous messaging does not preclude fast response times, as is required to support interactive applications. *Asynchronous messaging supports higher levels of scalability and supports scenarios where a response message may be sent minutes, hours or days after the initial request message.* Asynchronous messaging may be combined transparently with store-and-forward intermediaries.

In de praktijk blijkt dat er toch situaties bestaan waarin het synchroon bevestigen van een bericht voordeel biedt:

<aside class="example">
**Casusbeschrijving**

Bij uitwisseling tussen een GDI-voorziening en een grote afnemer worden door de verzendende partij aan de ontvangende partij in korte tijd zeer hoge volumes berichten aangeboden. Vanwege de eis op betrouwbaarheid is voor de uitwisseling gekozen voor het ebMS2 RM-profiel. Deze ebMS berichten bevatten per zending 1 functioneel bericht. Hierdoor is de overhead van het ebMS protocol voor de ontvangende partij relatief hoog.

De berichtuitwisseling in de keten was oorspronkelijk opgezet volgens asynchrone bevestiging. Bij de ketenintegratietest tussen beide partijen kwamen verwerkingsproblemen aan de ontvangende kant aan het licht. In onderling overleg tussen de ketenpartners is afgesproken om de ebMS2 uitwisseling zo in te stellen dat acknowledgements synchroon verstuurd worden. Hierna verliep de uitwisseling volgens de afgesproken eisen aan capaciteit, performance en betrouwbaarheid.
</aside>

### Voorwaarden voor toepassen van het synchrone bevestiging

- Scope:
  - Dit profiel is alleen geldig voor de Digikoppeling ebMS2 RM-profielen
- Aanleiding:
  - Door omvang van het volume van uitwisseling van berichten in beperkte tijd bestaan verwerkingsproblemen bij (een van beide) providers. Asynchrone uitwisseling van berichten binnen het ebMS profiel blijft de defaultmodus. Dus als de verwerking probleemloos verloopt is er geen reden om over te gaan op synchrone uitwisseling.
- Voorwaarde:
  - De Digikoppeling oplossing van beide partijen ondersteunen het instellen van SyncReplymode op mshSignalsOnly. Het instellen van deze mode kan dus niet eenzijdig worden opgelegd.

<aside class="note">
    Indien de berichtuitwisseling via een intermediary verloopt dient deze ook de SyncReplymode te ondersteunen om de synchrone communicatie tussen partijen mogelijk te maken.
</aside>

## EB016 Correlatie van berichten

Voor het correleren van berichten in ebMS2 spelen de elementen *Messageid*, *ConversationId* en *RefToMessageId* een rol. Het gebruik van de combinatie *MessageId* en *RefToMessageId* en het *ConversationId* dienen verschillende doelen. Deze 2 doelen zijn hieronder uitgewerkt:

### Response op een specifiek request

*Messageid* en *RefToMessageId* worden gebruikt om heel specifiek aan te geven op welke request een response wordt gegeven. *RefToMessageId* wordt dus gebruikt om de relatie tussen een ebMS bericht en het daarop volgende ACK of NACK bericht aan te geven, een protocol gerelateerde relatie.

<aside class="example">
Voor een terugmelding  op de GBA met MessageId 1234 moet de GBA in het antwoord op deze terugmelding het RefToMessageId 1234 gebruiken. Zo zijn request en response onlosmakelijk met elkaar verbonden. Normaal gesproken worden geen verdere antwoorden op een dergelijke melding terugverwacht waardoor het bericht met het RefToMessageId ook vaak de transactie beëindigt.
</aside>

De invulling van het RefToMessageId element voor "normale" messages (dwz met business documenten) is afhankelijk van de processpecificaties die in een domein worden afgesproken. Als daarin namelijk wordt besloten om de relatie tussen vraag en antwoord alléén tot uitdrukking te brengen in de business documenten zelf, bijvoorbeeld in een SBDH, dan is het mogelijk om het RefToMessageId in het ebMS bericht waarin het antwoord is opgenomen weg te laten.

### Response binnen een conversatie

Een *conversatie* gaat verder dan een simpele request/response. Een *conversation* is een gedefinieerd proces waarin meerdere berichten tussen twee partijen worden uitgewisseld. In alle ebMS berichten in een *instantie* van dat proces wordt dan hetzelfde *ConversationId* te gebruiken. De samenhang van de *individuele* transacties wordt bewaakt door het *MessageId* en *RefToMessageId* en de samenhang van het *proces* door het *ConversationId*. De waarde van 'ConversationId 'wordt in bepaald door de Partij die het eerste bericht verstuurd, de initiator.

Het gebruik van het ConversationId is dus op *business-niveau*, met verschillene implementaties:

- Simpel vraagbericht met ConversationId waarop het antwoordbericht hetzelfde ConversationId bevat
- Transactieachtig gebruik, waarbij meerdere vraag- en antwoordberichten, hetzelfde ConversationId bevatten
- Uitwisselingen waarin het case nummer in ConversationId wordt gezet, dat kunnen meerdere opeenvolgende transacties zijn waarbij er ook enige tijd kan verstrijken tussen de transacties
- Random, geen relatie tussen vraag en antwoordbericht op basis van ConversationId

<aside class="example">
Binnen het Justitie-domein is een goed voorbeeld hiervan het proces van de identiteitsvaststelling, waarbij één verzoek resulteert in meldingen naar meerdere registers waarna uiteindelijk een gecombineerd antwoord wordt teruggestuurd naar de Politie. Bij alle meldingen in één identiteitsvaststelling wordt hetzelfde ConversationId gebruikt, maar uiteraard hebben ze allemaal wel unieke MessageId’s (met bijbehorende RefToMessageId’s voor de antwoorden).
</aside>

In een losstaande transactie heeft het ConversationId dus niet zoveel toegevoegde waarde. Die toegevoegde waarde onstaat pas op het moment dat meerdere transacties door middel van het overkoepelende ConversationId met elkaar verbonden worden.

