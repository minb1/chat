# Technische foutmeldingen

## Categorieën

Bij gegevensuitwisseling kunnen er fouten optreden door verschillende oorzaken. Fouten kunnen in één van de volgende categorieën ingedeeld worden:

1. syntax fouten, hebben betrekking op de structuur van de berichten (XSD) en standaarden zoals WSA en SOAP

1. inhoudelijke fouten, hebben betrekking op inhoudelijke verwerking en zijn context/domein of sector specifiek en worden niet binnen DK gestandaardiseerd

1. protocolfouten, hebben betrekking op TLS of HTTP

1. fouten doordat een service niet (onvoldoende QoS) beschikbaar is, waaronder ook time-out en autorisatie problemen/fouten.

Per categorie kan op hoofdlijnen een procedure voor de foutafhandeling gedefinieerd worden.

1. bij syntax fouten dient zo mogelijk aangegeven te worden welk element fout is (zoals in foutmeldingen 0005 t/m 0008 aangegeven staat)

2. Zo mogelijk aangeven waarom. Bij inhoudelijke fouten aangeven dat het bericht vanwege inconsistentie niet verwerkt kan worden. (dit is eigenlijk geen transport/koppelvlak probleem, maar veeleer een business probleem met een bijbehorende afhandelingsprocedure, vgl. de terugmelding in het stelsel).

## Codes

Lijst van technische foutmeldingen met classificatie naar foutcategorieën:

| **Code** | **Omschrijving**                            | **Categorie** | **toelichting**                                                                                     |
|----------|---------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------|
| 0001     | Invalide soap envelope                      | 1             | Voldoet niet aan verwachte syntax. Structuur van de envelope matcht niet met wat er verwacht wordt  |
| 0002     | Niet geautoriseerd                          | 3             | Service niet beschikbaar (QoS). Door gebrek aan bevoegdheden.                                       |
| 0003     | Invalide soapaction                         | 1             | De inhoud leidt niet tot een voltooide actie, is niet gedefinieerd of onbegrijpelijk. Protocol fout |
| 0004     | Niet conform xsd                            | 1             | Voldoet niet aan verwachte syntax                                                                   |
| 0005     | WS-Addressing header `To` ontbreekt         | 1             | Ontbreekt of voldoet niet aan verwachte syntax                                                      |
| 0006     | WS-Addressing header `Action` ontbreekt     | 1             | Ontbreekt of voldoet niet aan verwachte syntax                                                      |
| 0007     | WS-Addressing header `MessageID` ontbreekt  | 1             | Ontbreekt of voldoet niet aan verwachte syntax                                                      |
| 0008     | WS-Addressing header `RelatesTo` ontbreekt | 1             | Ontbreekt of voldoet niet aan verwachte syntax                                                      |
| 0009     | Niet volgens UTF                            | 1             | Voldoet niet aan verwachte characterset                                                             |
| 0010     | Headers anders dan WSA-headers              | 1             | Voldoet niet aan verwachte syntax                                                                   |
| 0011     | Header andere waarde dan voorgeschreven     | 1             | Voldoet niet aan verwachte spec/waarde                                                              |
| 0051     | Service niet beschikbaar                    | 3             | Service niet beschikbaar (QoS). Door gebrek aan resources/ verwerkingscapaciteit.                   |
