## Message Ordering in ebXML

Een onderdeel van de ebMS 2.0 specificatie is de volgordelijkheid van berichten, aangeduid met MessageOrder. overgenomen uit hoofdstuk 9 van [[EBXML-MSG]]

### MessageOrder Module**

> The MessageOrder module allows messages to be presented to the To Party in a particular order. This is accomplished through the use of the MessageOrder element. Reliable Messaging MUST be used when a MessageOrder element is present.
> MessageOrder module MUST only be used in conjunction with the ebXML Reliable Messaging Module (section 6) with a scheme of Once-And-Only-Once (sections 6.6). If a sequence is sent and one message fails to arrive at the To Party MSH, all subsequent messages will also fail to be presented to the To Party Application (see status attribute section 9.1.1).

### MessageOrder Element**

> The MessageOrder element is an OPTIONAL extension to the SOAP Header requesting the preservation of message order in this conversation.

De ebMS standaard biedt daarmee de mogelijkheid om de volgordelijkheid van berichten te garanderen.

Maar het is wel een OPTIONAL<sup>6</sup> element, dus bekijk per product of het ook daadwerkelijk ondersteund wordt.

> <sup>6</sup>. OPTIONAL, uit [[EBXML-MSG]]: “This word means that an item is truly optional. One vendor may choose to include the item because a particular marketplace requires it or because the vendor feels that it enhances the product while another vendor may omit the same item.”

### Productondersteuning

De ondersteuning voor de MessageOrder verschilt per product. Hermes 2.0 en OrionMsg ondersteunen het wel, AxWay ondersteunt het niet, en de recente IBM release 'WebSphere Partner Gateway V6.1' ondersteunt het wel.

De Drummond Group voert jaarlijks ebXML interoperabiliteitstesten uit, waarmee leveranciers hun ebMS producten kunnen laten certificeren. Er wordt echter niet getest op MessageOrder.

Uit het test rapport van de Drummond Group, blz 18, hoofdstuk “Differing interpretations on the use of ConversationId”:

> (..) The ebMS v2.0 specification requires that ConversationId be present in all messages, and requires that if you implement the optional MessageOrdering feature (not tested by DGI) that ConversationId must stay the same over all ordered messages. (..)

### Zelfbouwoverwegingen

Als het niet mogelijk is om de MessageOrder functionaliteit te gebruiken, kan zelfbouw overwogen worden. Het is wel raadzaam om een aantal aspecten in overweging te nemen voordat de implementatie van de volgordelijkheid in een applicatie opgepakt wordt.

- Worden berichten die niet in volgorde verwerkt hoeven te worden onderscheiden van berichten die wel in volgorde verwerkt moeten worden? Door verschillende berichttypes te gebruiken kan er op eenvoudige wijze onderscheid gemaakt worden tussen berichtstromen waarin volgordelijkheid al dan niet van belang is. De achterliggende gedachte is, dat het niet noodzakelijk is om alle berichten in volgorde te verwerken.

- Is er een functionele behoefte aan bevestigingen? Zo ja, dan is volgordelijkheid niet van belang.

- Hoe vaak komt het voor dat de volgorde wel van belang is? Als dat incidenteel voorkomt, zou een ontvangstbevestiging retour kunnen gaan, waarna een volgend bericht verzonden mag worden.

- Er zullen afspraken gemaakt moeten worden om situaties te kunnen identificeren (en bijbehorende acties uit te voeren) als bijvoorbeeld één specifiek bericht niet aangekomen is, ook niet met behulp van de betrouwbare overdracht. De stroom van de te verwerken berichten “stokt” dan.

- Welke acties onderneemt de ontvangende applicatie om de verzendende applicatie hierover te informeren?

- Welke consequenties heeft dit voor verzendende partij?

- In hoeverre moet dit proces geautomatiseerd worden?

- Het inregelen van dit proces is lastig en het is dan de vraag of een andere oplossing (zoals met bevestigingsberichten) een goed alternatief is.

### Ontwerp Pattern

Als uitgangspunt voor de realisatie van de volgordelijkheid kan het Resequencer patroon gebruikt worden: [http://www.enterpriseintegrationpatterns.com/Resequencer.html](http://www.enterpriseintegrationpatterns.com/Resequencer.html)

<aside class="example">

A [Message Router](http://www.enterpriseintegrationpatterns.com/MessageRouter.html) can route messages from one channel to different channels based on message content or other criteria. Because individual messages may follow different routes, some messages are likely to pass through the processing steps sooner than others, resulting in the messages getting out of order. However, some subsequent processing steps do require in-sequence processing of messages, for example to maintain referential integrity.

*How can we get a stream of related but out-of-sequence messages back into the correct order?*

![Resequencer Pattern](media/resequencer_pattern.png "Resequencer Pattern")

*Use a stateful filter, a Resequencer, to collect and re-order messages so that they can be published to the output channel in a specified order.*

The Resequencer can receive a stream of messages that may not arrive in order. The Resequencer contains an internal buffer to store out-of-sequence messages until a complete sequence is obtained. The in-sequence messages are then published to the output channel. It is important that the output channel is order-preserving so messages are guaranteed to arrive in order at the next component. Like most other routers, a Resequencer usually does not modify the message contents.  

</aside>

De oplossingsrichtingen voor berichten waarvoor een volgnummer van belang is wordt hieronder globaal beschreven. Er wordt uitgegaan van een 'push' mechanisme: de ontvangende applicatie wordt dus actief doordat de ebMS adapter een functie van de applicatie aanroept voor het afleveren van een bericht (bijvoorbeeld met behulp van een web service of JMS queue). Dit in tegenstelling tot een 'pull' mechanisme waarbij het initiatief bij de applicatie ligt om te bepalen of er een nieuw bericht is ontvangen.

#### Specificatie (Design Time)

- Voeg aan de specificatie van het bericht een element 'Volgnummer' toe.

- Definieer een 'Aanvang' en en een 'Afsluit'-bericht waarmee de ontvangde partij geinformeerd wordt over de te verwerken berichten. Dit kan met name van belang zijn als er meerdere parallelle stromen van berichten zijn die ieder afzonderlijk gebruik maken van volgordelijkheid. Het is dan wel van belang de ConversationId te gebruiken.

- Indien gewenst kan er een bericht gedefinieerd worden waarmee de ontvangende partij de verzendende partij kan informeren over de verwerkings toestand.

- De applicatie moet bijhouden wat het volgnummer is van het laatst verwerkte bericht.

- Er is een 'berichtenpool' beschikbaar waar berichten met een volgnummer in bewaard worden. Hierbij is het volgnummer een sleutel om berichten uit de ‘berichtenpool' te halen.

#### Verwerking (Run Time)

- Bij ontvangst van een 'Aanvang'-bericht wordt de toestand geïnitieerd voor de volgordelijke verwerking van de berichten.

- Handel bij ontvangst van een bericht als volgt:

- Plaats het bericht in een ‘berichtenpool’.

- Als bericht nummer N verwerkt is, moet de applicatie bericht nummer N+1 ophalen uit de ‘berichtenpool’.

  - Als deze er niet is, zal de applicatie geen bericht verwerken.

  - Als deze er wel is, zal de applicatie het bericht verwerken EN daarna stap 2 opnieuw uitvoeren om een volgend bericht uit de ‘berichtenpool’ te verwerken.

Om te voorkomen dat een applicatie in een 'wait lock' terecht komt (één van de berichten in de sequentie komt niet aan, ook niet binnen de gestelde termijn van de betrouwbare overdracht), zal bekeken moeten worden wat de timingkarakteristieken zijn voor de verwerking van een volgend bericht.

Bij gebruik van het 'pull' mechanisme kan de berichtenpool gebruikt worden zoals in stap 2 beschreven: de applicatie zal dan op gezette tijden (op eigen initiatief) een bericht halen uit de berichtenpool. De ebMS adapter zal de berichten dan wel moeten afleveren aan de berichtenpool (zoals in stap 1).
