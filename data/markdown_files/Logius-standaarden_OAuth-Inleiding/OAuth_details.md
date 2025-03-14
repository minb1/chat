# OAuth details



Verdieping op Par 3.1.1 & Token exchange

 #### Token exchange

Where possible, the token exchange [[rfc8693]] grant type SHOULD be implemented instead of client credentials grant type, as this proves the identity of the user (and, where applicable, a second user that may act on behalf of the user). 

To Do add as a third flow in this document in usecases

Voorbeelden token exchange (rfc8693)

• Impersonation. Een achterliggende applicatie doet namens een andere applicatie een API aanroep met een tweede token (delegation scenario, met act claim). Het tweede token zal vaak minder of andere scopes of audience restricties hebben dan het originele token. Een ander bekend voorbeeld is dat het token slechts geldig is in de context van één transactie, en/of dat het token langer geldig is, bijvoorbeeld bij asynchrone (batch) verwerking van gegevens. 

• Delegation. Een gebruiker (vertegenwoordigd door een actor token) acteert namens een andere gebruiker (subject token).



@@@ [[rfc8693]] distinguishes impersonation and delegation scenarios.