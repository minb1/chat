
# Foutberichten

| Begrip            | Toelichting                                             |
|-------------------|---------------------------------------------------------|
|  Doel             | Aangeven dat er een fout is opgetreden.                 |
|  Voorwaarde       | Een foutbericht is altijd een reactie op een reeds verzonden bericht, dus één van de vraagberichten uit de voorgaande secties (AnnotatieToevoegenRequest, StatusoverzichtRequest, DetailsTerugmeldingRequest). |
|  Trigger          | Het verzonden vraagbericht leidt tot een fout in de verwerking ervan. |
|  Direct gevolg    | Een foutbericht wordt verstuurd. |
|  Vervolgactie     | De ontvanger van het foutbericht dient de fout af te handelen. |
|  Bijzonderheden   |                                  |

Een aantal foutsituaties is voorstelbaar. Er kan een probleem zijn met
autorisatie, met de syntax van de gegevens of er kan een fout optreden
in het systeem dat de annotatie moet verwerken.

## Autorisatiefouten

Wanneer een systeem van de afnemer op transportniveau of
applicatieniveau niet geautoriseerd is bij het leverende systeem dan
krijgt deze een Digikoppeling-foutmelding 'niet geautoriseerd' terug.

Optioneel kan de basisregistratie in de SOAP Fault-detail meer
informatie verstrekken.\
Indien een basisregistratie een eigen specifiekere foutcode heeft dan
mag deze ook verzonden worden in plaats van de Digikoppeling-melding.

## Verwerkingsfouten

In het geval dat een verwerkingsfout er toe leidt dat een antwoord
volledig uitblijft, bijvoorbeeld door het offline zijn van het
verwerkende systeem, dan zal de Digikoppeling-adapter van de afnemer een
time-out geven.

Indien het bevraagde systeem het bericht niet kan verwerken door een
systeemfout maar nog wel kan antwoorden, dan krijgt de afnemer een SOAP
Fault-bericht terug met de Digikoppeling-fout DK0051 (time-out),
optioneel kan de basisregistratie in de SOAP Fault-detail meer
informatie verstrekken. Wanneer het bericht niet verwerkt kan worden
vanwege een probleem met de inhoud van het bericht, anders dan de
syntax, dan wordt een DK0050-bericht ('kan bericht niet verwerken')
verstuurd. Indien een basisregistratie een eigen specifieke foutcode
heeft dan mag deze in plaats van de Digikoppeling-foutmelding verstuurd
worden.

## Syntaxfouten

Indien er een syntaxfout zit in de Digikoppeling-headers, dan volgt
hierop een SOAP Fault met de juiste foutcode volgens de
Digikoppelingstandaard (er bestaan specifieke foutcodes voor fouten in
headervelden).

Indien de inhoud van het vraagbericht niet voldoet aan de syntax van het
vraagbericht-XSD of een andere XSD die over de inhoud van het bericht
gaat dan wordt een SOAP Fault verstuurd; deze heeft als code DK0004
('element niet conform XSD'). In de SOAP Fault-detail wordt door de
basisregistratie aangegeven tegen welke XSD het bericht niet valideerde.
Optioneel is de basisregistratie vrij om nog meer informatie mee te
geven in de toelichting. Indien een basisregistratie een eigen
specifieke foutcode heeft dan mag deze in plaats van de
Digikoppeling-foutcode verstuurd worden.

Synchrone fouten voor Digikoppeling-WUS-berichten worden verstuurd als
SOAP Fault zoals dit is voorgesteld voor de berichtenstandaard van het
stelsel.<sup>6</sup> SOAP Faults worden doorgaans door de Digikoppeling-adapter
doorgestuurd naar de achterliggende applicatie. Dit koppelvlak gaat er
vanuit dat dit ook gebeurt voor de fouten die hier beschreven worden. De
foutafhandeling dient in de terugmeldapplicatie plaats te vinden. Binnen
de SOAP Fault wordt een aantal velden onderkend te weten:

<p class="note">
<sup>6</sup> Zie voorstel ["foutafhandeling synchrone berichten" van de "gemeenschappelijke afspraken berichtstandaarden"](https://digistandaarden.pleio.nl/groups/profile/24027452/gemeenschappelijke-afspraken-berichtstandaarden-gab)
</p>

|  |  |  |
|---------------------------------------------------------|----------|-------|
| **Berichttype**: DigimeldingSynchroonFault | *Synchrone foutafhandeling gaat middels een SOAP Fault* |           |
| faultcode        | *De plek waar de fout is opgetreden, één string opgebouwd uit de elementen \<Defaultwaarde\>.\<Code\>.\<OmschrijvingKort\> :*        | \[1..1\]  |
|                  | Defaultwaarde: *Bevat één van de defaultwaarden VersionMismatch, MustUnderstand, Client en Server.*<sup>7</sup> *Voor Digimeldingpraktijk zijn alleen Client- of Server-waarden relevant om aan te geven wat de aard van de fout is.* |           |
|                  | Code: *De specifieke foutcode die hoort bij de technische fout (format: \<afkortingbron\>\<codering van fout\>.*  |           |
|                                                         |           |
|                  | OmschrijvingKort: De korte omschrijving van de fout (bijvoorbeeld: de Digikoppeling omschrijving uit de lijst met foutmeldingen.* _Zie onderstaand voorbeeld._  |           |
| faultstring      | *De eigen meer gedetailleerde beschrijving van de foutsituatie.*<br>*De eigen beschrijving zoveel mogelijk geschikt maken voor het kunnen presenteren aan gebruiker.*<br>*De ontvanger is niet verplicht deze tekst over te nemen.*  | \[1..1\]  |
| faultactor       | *Bevat een URI van de antwoordende service.*<br>*Vul de faultactor in met de URI van de bron van de oorzaak, indien het SOAP-bericht langs een tussenstation gaat. Bijvoorbeeld Digimelding Webservice.*  | \[0..\*\] |
| faultdetail      | *Volledig vrij veld om nadere toelichting op de fout te geven, kan gebruikt worden om bijv. achterliggende applicatiefoutmeldingen mee te geven (xs:any).* | \[0..\*\] |

<p class="note">
<sup>7</sup> Zie <http://www.w3.org/TR/2000/NOTE-SOAP-20000508> sectie 4.4.1 voor uitleg
</p>

Voorbeeld:

<aside class="example" title="Foutcode"><pre>
<soap:Fault>
\<soap:Fault\>
  \<faultcode\>*soap:Server*\</faultcode\>
  \<faultstring\>*Fout*\</faultstring\>
  \<detail\>
    **<DigimeldingSynchroonFault** xmlns=\"\<http://webservices.digimelding.nl/dmks/cookiebox/\>\"\>
    \<faultcode\>*001*\</faultcode\>
    \<faultstring\>*Foutmelding*\</faultstring\>
    \<faultactor\>*Actor*\</faultactor\>
    \<faultdetail/\>
    **\</DigimeldingSynchroonFault\>**
  \</detail\>
\</soap:Fault\>
</pre></aside>

Generieke foutcodes van toepassing op Digimelding
-------------------------------------------------

Voor fouten op protocolniveau dienen de fouten van Digikoppeling gevolgd
te worden. Voor generieke fouten in Digimelding zijn de volgende
Digikoppeling-foutcodes van toepassing:

| **Nr** |  **Omschrijving**                           |   **Toelichting**       |
|--------|---------------------------------------------|-------------------------|
|  DK0002 |  Requester systeem niet geautoriseerd      | Indien je voor een basisregistratie niet geautoriseerd bent, ontvang je geen inhoudelijke informatie van die basisregistratie. Het autorisatiemodel wordt door de basisregistratie bepaald. In de foutmelding wordt de betreffende basisregistratie, waarvoor geen autorisatie bestaat, teruggegeven. |
|  DK0004 |  Element niet conform XSD                  | Bij een validatiefout wordt meegegeven in het SOAP Faultdetail aan welke XSD niet voldaan wordt. |
|  DK0050 |  Proces voor afhandelen bericht geeft fout | Wanneer een bericht zich niet houdt aan afspraken die gemaakt zijn rondom de vulling van de payload welke basisregistratiespecifiek zijn, bijvoorbeeld: Status is volgens de XSD een vrij tekstveld, iedere basisregistratie is vrij deze te kiezen. De basisregistratie kan teruggeven dat een status bij hen niet bekend is. |
|  DK0051 |  Antwoordend systeem geeft time-out        | Indien de achterliggende applicatie niet draait of een fout produceert dan kan het geen antwoord geven terwijl een geldig antwoord wel mogelijk zou moeten zijn en de messaging stack(Digikoppeling) nog wel in de lucht is. |
