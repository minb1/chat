# Digikoppeling profielen WUS

Op Digikoppeling wordt gewerkt met zogenaamde “profielen”. Een profiel is een
gedefinieerde bundeling van functionaliteit en daarmee van voorschriften. In de
huidige Digikoppeling versie zijn drie WUS-profielen aanwezig:

- Digikoppeling 2W-be,

- Digikoppeling 2W-be-S en

- Digikoppeling 2W-be-SE.

Voor de drie profielen geldt dat deze zowel geoptimaliseerd (volgens de MTOM
standaard) als niet geoptimaliseerd verstuurd kunnen worden. Hierom zijn voor
elk profiel voorbeeldberichten beschikbaar in zowel de geoptimaliseerde (MTOM)
als niet geoptimaliseerde vorm. Alle voorbeeldberichten zijn gebundeld in een
set [[[?Voorbeelden]]] en beschikbaar op de [[[?Logius]]].

Ten aanzien van beveiliging maken alle profielen gebruik van tweezijdig TLS, zie
hoofdstuk 2 paragraaf “Point-to-Point beveiliging”.

![Digikoppeling WUS profielen bevinden zich in de logistieke laag dat via een SOAP processor verbonden is met de applicatielaag.](media/wus_profielen_digikoppeling.png "Schematische weergave Digikoppeling WUS profielen")

In het bovenstaande figuur wordt schematisch weergegeven welke WUS profielen
Digikoppeling biedt en wat de algemene verschillen zijn. Alle profielen worden
in de volgende paragrafen in meer detail beschreven. In set [[[?Voorbeelden]]]
zijn voorbeeld berichten opgenomen voor de drie profielen. Deze bieden een
duidelijk inzicht hoe de berichten uiteindelijk eruit komen te zien.

## WUS Profiel Digikoppeling 2W-be

<dl>
<dt>Beveiliging</dt>
<dd>
Dit profiel maakt voor de beveiliging alleen gebruik van tweezijdig TLS.
</dd>
<dt>Headerblocks</dt>
<dd>
Alleen de verplichte WS-Addressing headers zijn hier van toepassing (zie
soapenv:Header in [[[?Voorbeelden]]]).
</dd>
<dt>MTOM</dt>
<dd>
De geoptimaliseerde (MTOM) voorbeeldberichten worden weergegeven in [[[?Voorbeelden]]].
</dd>
</dl>

## WUS Profiel Digikoppeling 2W-be-S 

<dl>
<dt>Beveiliging</dt>
<dd>
Dit profiel maakt voor de beveiliging gebruik van tweezijdig TLS en tevens
worden de berichtonderdelen ondertekend zoals vermeld in hoofdstuk 2, paragraaf
End-to-End beveiliging.
</dd>
<dt>Headerblocks</dt>
<dd>
In dit profiel zijn de WS-Addressing en WS-Security 1.0 ondertekening
(wsse:Security) headers van toepassing (zie [[[?Voorbeelden]]]).
</dd>
<dt>MTOM</dt>
<dd>
De geoptimaliseerde (MTOM) voorbeeldberichten voor profiel Digikoppeling 2W-be-S
worden weergegeven in [[[?Voorbeelden]]].
</dd>
</dl>

## WUS Profiel Digikoppeling 2W-be-SE

<dl>
<dt>Beveiliging</dt>
<dd>
Dit profiel maakt voor de beveiliging gebruik van tweezijdig TLS en tevens
worden de berichtonderdelen ondertekend en versleuteld zoals vermeld in
hoofdstuk 2, paragraaf End-to-End beveiliging.
</dd>
<dt>Headerblocks</dt>
<dd>
In dit profiel zijn de WS-Addressing en WS-Security 1.0 ondertekening
(wsse:Security) headers van toepassing. Ook wordt hierbij de payload van het
bericht versleuteld (xenc:EncryptedData) (zie [[[?Voorbeelden]]]).
</dd>
<dt>MTOM</dt>
<dd>
De geoptimaliseerde (MTOM) voorbeeldberichten voor profiel Digikoppeling
2W-be-SE worden weergegeven in [[[?Voorbeelden]]].
</dd>
</dl>
