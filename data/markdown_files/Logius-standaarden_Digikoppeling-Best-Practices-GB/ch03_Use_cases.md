# Use cases

Bij de definitiestudie naar Digikoppeling Grote Berichten zijn een aantal use cases onderkend die bij de uitwerking van de standaard als referentie hebben gediend. Deze use-cases zijn bij toepassing van de standaard nuttig als inspiratiebron voor vormgeving van uitwisselingspatronen.

Dit hoofdstuk beschrijft deze use-cases en geeft aan hoe deze gerealiseerd kunnen worden. Daarbij maakt de uitwerking duidelijk of gegevens via REST-API, ebMS2, WUS en/of als groot bericht verstuurd worden. Beschrijving van de use-cases is generiek; specifieke implementaties kunnen variëren maar zullen vaak deze generieke patronen als basis hebben.

## Download

In deze use case is een benodigde gegevensset al beschikbaar in een download-server bij de service provider. De client dient echter nog de beschikking te krijgen over een verwijzing naar de gewenste gegevensset. Daartoe worden de volgende processtappen doorlopen:

![Use case 'Download'](media/use_case_download.svg "Use case 'Download'")

1. De service requester / client bevraagt (REST-API, WUS of ebMS2 request) de service provider met kenmerkende criteria voor de gezochte gegevensset. Eventueel kan deze bevraging in enkele tussenstappen verlopen als deze initieel te veel mogelijkheden oplevert.
2. De service provider levert een verwijzing naar de gezochte gegevens set in de vorm van Meta-data (WUS-response of ebMS2).
3. De service requester haalt de gewenste gegevensset op (Groot Bericht) en krijgt deze op grond van autorisatie (OIN).

## Selectie

In deze use case vraagt een service requester een gegevensselectie van een service provider. Daartoe worden de volgende processtappen doorlopen:

![Use case 'Selectie'](media/use_case_selectie.svg "Use case 'Selectie'")

1. De service requester bevraagt (ebMS2-request of eventueel REST-API of WUS) de service provider met kenmerkende criteria voor de gezochte gegevensset. Eventueel kan deze bevraging in enkele tussenstappen verlopen als deze initieel te veel mogelijkheden oplevert.
1. De service provider maakt de gewenste gegevens set aan en zet deze klaar in een Groot Bestand.
1. Zodra het Grote Bestand gereed is, Meldt de service provider dit aan de eerdere service requester met een verwijzing naar de gezochte gegevens set in de vorm van Meta-data (ebMS2).
1. De service requester haalt de gewenste gegevensset op (Groot Bericht) en krijgt deze op grond van autorisatie (OIN).

Merk op dat deze use case vrijwel gelijk is aan “Download”. Alleen stap 2 'aanmaken selectie' is aanvullend. Vanwege de benodigde tijd van dit aanmaken kan gereedmelding niet via een WUS-response plaatsvinden en zal altijd ebMS2 toegepast moeten worden. Als het 'aanmaken van de selectie binnen de time-out van het WUS-request kan plaatsvinden ontstaat als vanzelf het “Download” pattern.

## Verzending (pull)

In deze use case verzendt een service provider gegevens naar een service requester. Daartoe worden de volgende processtappen doorlopen:

![Use case 'Verzending'](media/use_case_verzending.svg "Use case 'Verzending'")

1. Op enig moment is er een gebeurtenis waardoor de service provider besluit tot verzenden van gegevens. Voorbeelden van triggers zijn: tijd, bericht ontvangst of wijziging van gegevensobjecten.
2. Als gegevens niet beschikbaar zijn in de vorm van een Groot Bestand zal dit aangemaakt worden (bijvoorbeeld door samenstelling vanuit een database).
3. De service provider stuurt een 'verzoek tot ophalen' naar de service requester. In dit verzoek is in ieder geval de Meta-data van het Grote Bestand opgenomen, maar kan bijvoorbeeld ook informatie over de aard en beoogde afhandeling van het Grote Bestand zijn opgenomen.
4. De service requester haalt de gegevens op (Groot Bericht) en krijgt deze verstrekt op grond van autorisatie (OIN).

> Aanvullend kan het Grote Bestand ook verwijderd worden. Bijvoorbeeld nadat de expiration-time is verstreken of nadat de client een bericht heeft gestuurd om de succesvolle ontvangst te bevestigen. Merk op dat deze interactie identiek is aan “Multi-distributie” maar slechts één (1) afnemer kent.

## Push

In deze use case verzendt een service provider gegevens naar een service requester. Daartoe worden de volgende processtappen doorlopen:

![Use case 'Push'](media/use_case_push.svg "Use case 'Push'")

Op enig moment is er een gebeurtenis waardoor de service provider besluit tot verzenden van gegevens. Voorbeelden van triggers zijn: tijd, berichtontvangst of wijziging van gegevensobjecten.
1. De service requester stuurt een 'upload verzoek' naar de service provider. In deze melding is in ieder geval de Meta-data van het Grote Bestand opgenomen, maar kan bijvoorbeeld ook informatie over de aard en beoogde afhandeling van het Grote Bestand zijn opgenomen.
2. De service provider stuurt een 'upload response' met instructies over de uploadlocatie (UUID).
3. De service requester upload de gegevens set (Groot Bericht) en krijgt toegang op grond van autorisatie (OIN).

> Merk op dat deze interactie overeenkomsten vertoont met de use case “Verzending” maar upload in plaats van download.

## Multi-distributie

In deze use case distribueert een service provider gegevens naar meerdere Afnemers. Daartoe worden de volgende processtappen doorlopen:

![Use case '(multi-) distributie'](media/use_case_multidistributie.svg "Use case '(multi-) distributie'")

1. Op enig moment is er een gebeurtenis waardoor de service provider besluit tot distributie van gegevens. Voorbeelden van triggers zijn: tijd, berichtontvangst of wijziging van gegevensobjecten.
1. Als gegevens niet beschikbaar zijn in de vorm van een Groot Bestand zal dit aangemaakt worden (bijvoorbeeld door samenstelling vanuit een database).
1. De service provider stuurt een Melding (ebMS2) naar de Afnemers die deze gegevens (bijvoorbeeld op basis van een abonnement) behoren te ontvangen. In deze Melding is in ieder geval de Meta-data van het Grote Bestand opgenomen, maar kan bijvoorbeeld ook informatie over de aard en beoogde afhandeling van het Grote Bestand zijn opgenomen. Alle Afnemers ontvangen dezelfde Meta-data.
1. Afnemers halen de gegevens op (Groot Bericht) en krijgen deze verstrekt op grond van autorisatie (OIN).

Aanvullend kan het Grote Bestand ook verwijderd worden. Bijvoorbeeld nadat de expiration-time is verstreken of nadat alle Afnemers een bericht hebben gestuurd om de succesvolle ontvangst te bevestigen.

## Upload

In deze use case upload een client naar een service provider. Voor het voorbeeld is Digipoort gebruikt als service provider en met behulp van het Push principe wordt met behulp van een REST-API koppeling de metadata gedeeld. Daartoe worden de volgende processtappen doorlopen:

![Use case 'Upload'](media/use_case_upload.svg "Use case 'Upload'")

1. Op enig moment is er een gebeurtenis waardoor de service requester besluit tot verzenden van gegevens. Voorbeelden van triggers zijn: tijd, berichtontvangst of wijziging van gegevensobjecten.
2. Als de gegevensset nog niet beschikbaar is in de vorm van een Groot Bestand zal dit aangemaakt worden (bijvoorbeeld door samenstelling vanuit een database)
3. De service requester maakt gebruik van een HTTP POST operatie om de metadata van de gegevens set aan de service provider (Digipoort) te sturen. De service requester krijgt hiervoor authorisatie op grond van het HRN in het PKIO certificaat.
4. De service requester upload de gegevens set (Groot Bericht) eventueel gebruikt hij in de upload de unieke referentie die is teruggegeven bij het creeren van de resource met de HTTP POST.
5. De service requester maakt gebruik van een HTTP GET operatie om de status van de upload op te vragen bij de gecreeerde resource.

## Business voorbeelden

| **Use case**        | **Voorbeeld**                                                                                 |
| ------------------- |-----------------------------------------------------------------------------------------------|
| upload              | Clients versturen een bericht en bijbehorend Groot Bestand.                                   |
| push                | Clients uploaden een Groot Bestand na goedkeuring van het upload verzoek.                     |
| download            | Gegevens worden beschikbaar gesteld voor breed (openbaar) gebruik.                            |
| selectie            | Ad hoc informatieverzoeken voor het samenstellen van rapportages of doorsnedes van gegevens . |
| verzending          | Maandelijkse verzending van vast afgesproken gegevenssets.                                    |
| (multi-)distributie | Verstrekking van mutaties naar geabonneerde instanties.                                       |
