# Het gebruik van berichtvolgordelijkheid

Digikoppeling Koppelvlakstandaard ebMS2 raadt het gebruik van volgordelijkheid van berichten sterk af. Reden hiervoor is dat niet elk product dat ebMS2 implementeert de volgordelijkheid ondersteunt (in ebMS2 wordt dit MessageOrder genoemd). Voor situaties waar beide partijen bi-lateraal over een product beschikken dat dit ondersteund, wordt het gebruik ervan voor het Digikoppeling ebMS2 Koppelvlak wel toegestaan. Voorwaarde daarbij is dat vooraf bekend is dat alle (ook toekomstige) communicatiepartners dit kunnen ondersteunen. Gebruik lijkt daarom alleen realistisch voor bilaterale situaties of een zeer beperkt aantal vooraf bekende communicatiepartners. Gebruik door landelijke voorzieningen zoals basisregistraties is onwenselijk.

Partijen die gebruik willen maken van de volgordelijkheid zullen daarom vooraf onderling moeten verifiëren of de functionaliteit door de andere partij ondersteund wordt en of de ebMS-adapters op dit onderdeel ook interoperabel zijn. (Het moge duidelijk zijn dat als beide partijen van hetzelfde product gebruik maken, de interoperabiliteit gegarandeerd zou moeten zijn.)

Het heeft tot gevolg dat veel partijen deze functionaliteit niet kunnen gebruiken. In deze situaties is volgordelijkheid op applicatieniveau een beter alternatief. Dit heeft als extra voordeel dat niet alleen zekeerheid bestaat over het in volgorde ontvangen, maar ook over het in volgorde verwerken van berichten.

De granulariteit (zie hieronder) van de berichten waarop volgordelijkheid toegepast moet worden en de verwerking in de organisatie zal nader bekeken moeten worden. De volgende twee hoofdstukken geven hierover enkele overwegingen.

## Granulariteit

Granulariteit betekent letterlijk: (fijn)korreligheid, ofwel de mate van detaillering. De granulariteit van de berichten waarvoor volgordelijkheid van belang is zal verstandig moeten worden gekozen.

De uitersten worden hieronder beschreven:

- Alle berichten die op basis van een ebMS2 contract (CPA) verzonden worden zijn van elkaar afhankelijk ten aanzien van de volgorde.

  - Als één enkel bericht faalt in de overdracht, heeft dit tot gevolg dat de gehele berichtenstroom stokt. Deze berichtenstroom kan dan pas weer op gang gebracht worden als het falende bericht opnieuw gesynchroniseerd is.

  - Bij grote hoeveelheden berichten die in een kort tijdsbestek verzonden worden zullen aan de ontvangende kant tijdelijk bewaard moeten worden: dit legt een claim op de resources van de ebMS-adapter.

  - Berichten die mogelijk een hogere prioriteit hebben kunnen niet eerder verwerkt worden dan passend is ten aanzien van de volgorde van reeds verzonden berichten.

- Voor elk afzonderlijk bericht wordt gekeken of het onderdeel uitmaakt van een nieuwe berichtenstroom waarvoor volgordelijkheid van belang is.

- Er kunnen meerdere berichtstromen actief zijn die onafhankelijk van elkaar de volgordelijkheid ondersteunen.

- Per object moet aangegeven worden (of bekend zijn) of de message order van belang is.

De granulariteit wordt in essentie bepaald door het object waarover berichten uitgewisseld worden en waarvan de volgordelijkheid van belang is: het gaat dan om transactionele berichten over hetzelfde object.

## Verwerking in de organisatie

Als de berichten op Digikoppeling in volgorde moeten worden afgeleverd, zal dit tot gevolg hebben dat diezelfde berichten ook in de juiste volgorde aangeboden worden aan de verwerkende applicatie in de organisatie.

## Alternatieven voor berichtvolgorde

Het garanderen van de volgordelijkheid binnen een keten kan in zijn algemeenheid op meerdere manieren worden geregeld:

1. Vermijden van de behoefte  
    Een zuivere gebeurtenis-gedreven architectuur kan de behoefte aan volgordelijkheid vaak vermijden. In deze architectuur worden gegevens niet meegeleverd met gebeurtenissen, maar naar aanleiding van gebeurtenissen bij de bron geraadpleegd. Als in een dergelijke situatie bijvoorbeeld overlijden eerder dan de geboorte doorgegeven wordt zal de actie die op overlijden (of geboorte) volgt altijd de meest actuele gegevens opleveren.

1. Risico-reductie  
    Bewuste vertragingen aan de bron tussen twee opeenvolgende (gerelateerde) gebeurtenissen, kan het risico op het uit volgorde raken van berichten beperken.

1. Applicatief: mitigatie  
    In deze situatie verwerkt de afnemer gebeurtenissen zodanig dat eventuele volgordeproblemen worden gemitigeerd (simpelste algoritme: als het BSN binnenkomt voordat de persoon bekend is, wordt het BSN terzijde gelegd totdat de persoon wél bekend is).

1. Applicatief: unipotente operaties  
    In deze situatie wordt er voor gezorgd dat de operatie naar aanleiding van een bericht slechts op één manier interpreteerbaar is. Bijvoorbeeld door bij verandering van een subsidie of burgelijke staat zowel de oude als de nieuwe situatie mee te geven. Of bij ‘saldo-informatie’ een verhoging of verlaging te sturen in plaats van de nieuwe waarde.

1. Applicatieve volgorde door ontvanger  
    In deze situatie geeft de applicatie aan het bericht informatie mee waarmee de ontvanger de volgorde kan bepalen. Dit kan bijvoorbeeld met tellers, zodat de applicatie de berichten voor verwerking in volgorde kan plaatsen, maar ook met timestamps zodat de applicatie na verwerking eventuele correcties kan uitvoeren (zie ook bijlage 1 onder “zelfbouwoverwegingen”). Dit zal liefst selectief alleen voor kritische berichten gebeuren zodat andere berichten ongestoord doorgang vinden.

1. Applicatieve volgorde door verzender  
    In deze situatie wacht de verzendende applicatie op een ontvangstbevestiging of verwerkingsbevestiging van de ontvanger voordat een nieuw bericht gestuurd wordt. Dit zal liefst selectief alleen voor kritische berichten gebeuren zodat andere berichten ongestoord doorgang vinden.

1. Logistieke volgorde  
    Digikoppeling biedt als optie om berichten in volgorde af te leveren aan de ontvangende applicatie. Deze functie van de Digikoppeling-adapter software wordt echter door enkele belangrijke leveranciers niet ondersteund. Digikoppeling stelt daarom dat deze optie alleen gebruikt kan worden als vóóraf bilateraal overeenstemming bereikt is over ondersteuning.
