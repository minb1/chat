# Profile Requirements Details

## Module: Core Extension Elements

### Profile Requirement Item: PartyId

|[[EBXML-MSG]] Section 3.1.1.1 | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|PartyId Element| Header elements:<br>`SOAP:Header/eb:MessageHeader/eb:From/eb:PartyId`<br>`SOAP:Header/eb:MessageHeader/eb:To/eb:PartyId`|
| Is a specific standard used for party identification? Provide details. Example - EAN•UCC Global Location Number. Ref.: ISO6523 - ICD0088. | Partners who are going to use ebMS for the first time **must** use an **OIN** (Organisatie Identificatie Nummer) for identification. Partners who are already using ebMS and are using other identification schemes are allowed to use their identification: the type attribute must identify their identification scheme and must be different from urn:osb:oin. The use of their own identification should be temporary: the partner should start using OIN at a certain moment for identification using Digikoppeling. For non-production environments a suffix is allowed after the OIN to distinguish it from production (e.g. “_OTA” or “_T”) <br><br> **OIN** stands for Organisatie Identificatie Nummer and is maintained by Logius in the *COR (Centrale OIN Raadpleegvoorziening)*. The number is unique and allows identification of partners, even if they are not themselves legal entities, but departments or units of larger organizations. <br><br> The OIN used for PartyId must be the same as the OIN from the end-party and should not contain the OIN from an intermediate party. In case the end-party is the same party that performs TLS, signing and/or encryption the OIN used for PartyId should be identical to the OIN used for the TLS-, signing- and/or encryption-certificate respectively. Hence, if the end-party does not perform TLS, signing and/or encryption the corresponding OIN’s may differ. |
| Should multiple PartyId elements be present in From and To elements? | (empty)  |
| Is the type attribute needed for each PartyId, and if so, what must it contain? Example – within the EAN•UCC system, the PartyId element and type are represented using Global Location Number.<span style='font-size:10pt'>`<eb:PartyId eb:type="http:/www.iso.int/schemas/eanucc/gln"/>1234567890128</eb:PartyId>`</span>| The type attribute **must** be present and **should** have the fixed value. <br>The following type attribute value has to be used in case of an OIN is used by the partner: **urn:osb:oin** |
|alignment|appears as PartyId element in CPA. (c)<br> appears as PartyId/\@type in CPA |
|Test References|(empty)|
|notes|ISO 6523 is an international standard registry of agencies issuing codes. Value 0106 in this registry identifies the Association of Chambers of Commerce and Industry in the Netherlands. The prefix urn:oasis:names:tc:ebxml-cppa:PartyId-type is used to indicate the issuing agency is an ISO 6523 registered agency. The type attribute allows unique identification of the agency that issues the number or code that identifies the partner. In theory, this mechanism allows multiple identification systems to be used in parallel, with no requirement that the codes in those systems do not overlap.|

### Profile Requirement Item: Role

|[[EBXML-MSG]] Section 3.1.1.2 Role Element| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Header elements|`/SOAP:Header/eb:MessageHeader/eb:From/eb:Role`<br>`/SOAP:Header/eb:MessageHeader/eb:To/eb:Role`|
| Are Roles defined for each party of each business process? List them, or provide a reference to the source of these values. Example – within the EAN•UCC system, approved values are specified by the EAN•UCC Message Service Implementation Guide. | Business process is out of scope for (this version of the) Digikoppeling. Within a single contract (CPA) between two Partners: - A Partner **must** fulfill one and only one role (a Partner cannot change its role within one contract). - A Partner can send messages (one or more) and/or receive messages (one or more). In case a Partner wants to use different roles, different contracts (CPA's) must be used. |
| Alignment | [Per-process; may reference Role values in BPSS [BPSS] definitions. Appears as Role/\@name in CPA.] |
| Test References | (empty)  |
| Notes  | (empty)  |

### Profile Requirement Item: CPAId


|[[EBXML-MSG]] Section 3.1.2 CPAId Element | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Header/eb:MessageHeader/eb:CPAId`|
| What identification scheme is used for the CPAId, and what form should it take? If it is a URI, how is it constructed? Does it reference a real CPA, or is it just a symbolic identifier? Example – within the EAN•UCC system, the value of the CPAId is the concatenation of the Sender and Receiver GLNs followed by a four digit serial number. 1234567890128 - GLN Party A 3456789012340 - GLN Party B 0001 - CPA Number between parties A and B | The proposed EAN•UCC is recommended as a good practice. |
| Alignment | Appears as CollaborationProtocolAgreement/\@cpaid in CPA.  |
| Test References | (empty)  |
| Notes  | (empty)  |

### Profile Requirement Item: ConversationId

|[[EBXML-MSG]] Section 3.1.3 ConversationId Element| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Header/eb:MessageHeader/eb:ConversationId`|
| What is the user definition of a Conversation? What is the business criterion used to correlate messages considered parts of the same conversation?   | [EBXML-MSG] requires that request messages, response messages, and any acknowledgments and error messages have the same value for ConversationId. |
| In case the MSH implementation gives exposure of the ConversationId as it appears in the header, what identification scheme should be used for its value, and what format should it have? If it is a URI, how is it constructed? In case the ConversationId is not directly exposed, but only a handle that allows applications to associate messages to conversations, if the value of this handle is under control of the application, what format should it have? | No recommendation made.  |
 If BPSS is used, ConversationId typically maps to a business transaction. Is that the case? Does it map to a business collaboration instead? | No recommendation made. Business process is out of scope for Digikoppeling. |
| Test References |(empty) |  |
| Notes  | ConversationId is a required ebXML message header element. |

### Profile Requirement Item: MessageId

|[[EBXML-MSG]] Section 3.1.6.1 | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Header/eb:MessageHeader/eb:MessageData/eb:MessageId`|
| Although there is no requirement for an MSH to give control about MessageId to an application, some implementations may allow this. In this case, is there any requirement on the source of this ID? Any length and format restrictions when the ID is generated? | No recommendation made. The value of MessageId does not need to meet any requirements beyond the string format specified in [EBXML-MSG] and the global uniqueness constraint of [[rfc5322]]. |
| Alignment | (empty) |
| Test References | (empty)|
| Notes  | (empty)|

### Profile Requirement Item: Service

|[[EBXML-MSG]] Section 3.1.4 Service Element| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Header/eb:MessageHeader/eb:Service`<br>`/SOAP:Header/eb:MessageHeader/eb:Service/\@type`|
| Are Services (related groups of Actions) defined for each party of each business process? List them, or provide a reference to the source of these values. [Per-process; absent from BPSS definitions.] Is there a URI format scheme for this element? | No recommendation made.   |
| Is there a defined "type" for Service elements? If so, what value must the type attribute contain?  | The text content of the Service element **must not** contain white space. |
| Alignment | Appears as Service element in CPA Appears as Service/\@type in CPA |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Action

|[[EBXML-MSG]] Section 3.1.5 Action Element| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Header/eb:MessageHeader/eb:Action`|
| Are actions defined for each party to each business process? List them, or provide a reference to the source of these values. [Per-process; may reference BusinessAction values in BPSS definitions. Example – within the EAN•UCC system, approved values are specified by the EAN•UCC Message Service Implementation Guide. \<eb:Action\>Confirmation\</eb:Action\> | No recommendation made. |
| Alignment | Appears as ThisPartyActionBinding/\@action in CPA.]   |
| Test References | (empty) |
| Notes  | The text content of the Action element in the header **must not** contain white space. |

### Profile Requirement Item: Timestamp (removed)

This item is no longer required.

### Profile Requirement Item: Description

|[[EBXML-MSG]] Section 3.1.8 Description Element | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Header/eb:MessageHeader/eb:Description` |
| Are one or more Message Header Description elements required? In what language(s)? Is there a convention for its contents? | No recommendation made. Description elements are not required. Message handlers **may** ignore Description elements. |
| Alignment |(empty) |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Manifest

|[[EBXML-MSG]] Section 3.2.2 Manifest Validation| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Body/eb:Manifest`|
| How many Manifest elements must be present, and what must they reference? Does the order of Manifest elements have to match the order of the referenced MIME attachments? Any restriction on the range of value for xlink:reference (e.g. nothing other than content id references)? | Manifest elements **must** only reference business documents or other payloads that are included in the ebXML message as a MIME part allows for references to external message payloads (for instance, using HTTP URIs), which are logically part of the message, but not as a physical entity in the MIME envelope. This is **never used** in these profiles. |
| Must a URI whichcannot be resolved be reported as an error?  | A Content Id URI reference that cannot be resolved **must** be treated as an error. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  |  XML or other business documents can have references to other resources that are not part of the ebXML message. It is up to the receiving application to interpret any such references. |

### Profile Requirement Item: Reference

|[[EBXML-MSG]] Section 3.2.1 Reference Element| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Body/eb:Manifest/eb:Reference`|
| Is the xlink:role attribute required? What is its value? | **Not applicable**. The xlink:role attribute is not required.   |
| Are any other namespace-qualified attributes required?   | **Not applicable**. No other namespace-qualified attributes are allowed. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | Only the Content Id reference mechanism [[rfc2392]] is allowed. |

### Profile Requirement Item: Reference/Schema

|[[EBXML-MSG]] | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Body/eb:Manifest/eb:Reference/eb:Schema`|
| Are there any Schema elements required? If so, what are their location and version attributes?  | Schema elements are not required. Digikoppeling does not perform XML schema validation. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Reference/Description

|[[EBXML-MSG]] Section 3.2.1.2 Description Element | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Body/eb:Manifest/eb:Reference/eb:Description`|
| Are any Description elements required? If so, what are their contents?   | Description elements are optional. They **may** be ignored by any receiving message service handler. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

Module: Security
----------------

### Profile Requirement Item: Signature generation

|[[EBXML-MSG]] Section 4.1.4.1 Persistent Digital Signature| **Best effort**<br>**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|
|Header elements:<br>SOAP:Header/Signature|||
| (a) Must messages be digitally signed? [Yes, for Security Services Profiles 1, 6-21.]  | ****Not applicable****. These profiles do not support XML Digital Signatures at the message handler level.| **Required** in this profile.   |
| Are additional Signature elements required, by whom, and what should they reference?  | **Not applicable**.   | **Never used** in this profile. |   |
| What canonicalization method(s) must be applied to the data to be signed? | **Not applicable**.   | The use of XML canonicalization is **required**. [[xml-exc-c14n]] |
| What canonicalization method(s) must be applied to each payload object, if different from above?  | **Not applicable**.   | **Not applicable**.  |
| What signature method(s) must be applied?   | **Not applicable**.   | The applied signature method is described in [[[DK-beveiliging]]]  |   |
| What Certificate Authorities (issuers) are allowed or required for signing certificates? | **Not applicable**.   | The use of PKI Overheid certificates is **required** in which an OIN is used in the Subject.serialNumber. [[[DK-beveiliging]]] |   |
| Are direct-trusted (or self-signed) signing certificates allowed?   | **Not applicable**.   | This profile is **never used**. Only used in testing and Proof of Concept environments   |   |
| What certificate verification policies and procedures must be followed?   | The requirements as stated in [[[PKIO-PvE]]] have to be used. The use of certificate revocation lists (CRL) from the trusted CAs is required.   |The requirements as stated in [[[PKIO-PvE]]] have to be used. The use of certificate revocation lists (CRL) from the trusted CA's is required.|
| Alignment|(a) Appears as BusinessTransactionCharacteristics/\@isAuthenticated=persistent and BusinessTransactionCharacteristics/\@isTamperProof=persistent in CPA | |
| Test References | (empty)  | (empty)  |
| Notes  | Applications submitting data to, or receiving data from, Digikoppeling ebXML Message service handlers can perform signing at the message payload level. The ebXML Messaging protocol is payload-neutral and therefore supports signed payloads. In that case, the Digikoppeling is not aware of the presence of signatures and does not perform signature verification. | for more information see [[[DK-beveiliging]]] |   |

### Profile Requirement Item: Persistent Signed Receipt

|[[EBXML-MSG]] Section 4.1.4.2 Persistent Signed Receipt| **Best effort**<br>**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|
|Header elements:<br>`/SOAP:Header/eb:Signature`|||
| Is a digitally signed Acknowledgment Message required? [Yes, for Security Services Profiles 7, 8, 10, 12, 14, 15, 17, 19-21. See the items beginning with Section 4.1.4.1 for specific Signature requirements.] | **Not applicable**.  | Signing acknowledgements is **required**. |
| If so, what is the Acknowledgment or Receipt schema?   | **Not applicable**.  | [[xmldsig-core-20020212]]  |   |
| Alignment | Appears as BusinessTransactionCharacteristics/\@isNonRepudiationReceiptRequired=persistent in CPA.  | |
| Test References | (empty) |(empty) |
| Notes  | (empty) |(empty) |

### Profile Requirement Item: Non Persistent Authentication

|[[EBXML-MSG]]Section 4.1.4.3 Non Persistent Authentication | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Are communication channel authentication methods required? [Yes, for Security Services Profiles 2-5.] Which methods are allowed or required? | Client and Server authentication is required using HTTPS and TLS. The currently allowed protocol versions for TLS are described in [[[DK-beveiliging]]] `Note:` Message service handlers **should NOT** be able to operate in SSL v3 backward compatibility mode. |
| Alignment | [Appears as BusinessTransactionCharacteristics/\@isAuthenticated=transient in CPA.] |
| Test References | (empty) |
| Notes  | for more information see [[[DK-beveiliging]]]  |

### Profile Requirement Item: Non Persistent Integrity


|[[EBXML-MSG]] Section 4.1.4.4 Non Persistent Integrity| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Header/eb:Signature`|
| Are communication channel integrity methods required? [Yes, for Security Services Profile 4.] Which methods are allowed or required? | **Not applicable**  |
| Alignment | [Appears as BusinessTransactionCharacteristics/\@isTamperproof=transient in CPA.] |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Persistent Confidentiality

|[[EBXML-MSG]] Section 4.1.4.1 Section 4.1.4.5 Persistent Confidentiality| **Best effort**<br>**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|
|Header elements:<br>`/SOAP:Header/eb:Signature`|||
| Is selective confidentiality of elements within an ebXML Message SOAP Header required? If so, how is this to be accomplished? [Not addressed by Messaging Specification 2.0.] | **Not applicable**.  | **Not applicable**. | Is payload confidentiality (encryption) required? [Yes, for Security Services Profiles 13, 14, 16, 17, 21, 22.] Which methods are allowed or required?   | **Not applicable**.  | Payload confidentiality is **optional**. [[[DK-beveiliging]]] describes what security standard must be used. |
| Alignment | [Appears as BusinessTransactionCharacteristics/\@isConfidential=persistent in CPA.]   |  
| Test References | (empty)| (empty)  |
| Notes  | Applications submitting data to, or receiving data from, Digikoppeling message handlers can perform encryption at the payload processing level. The ebXML Messaging protocol is payload-neutral and therefore supports transport of encrypted payloads. However, any encryption and decryption of payloads is out of scope for these profiles. |

### Profile Requirement Item: Non Persistent Confidentiality

|[[EBXML-MSG]] Section 4.1.4.6 Non Persistent Confidentiality| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Header/eb:Signature`|
| Are communication channel confidentiality methods required? [Yes, for Security Services Profiles 3, 6, 8, 11, 12.] Which methods are allowed or required? | The use of HTTPS and TLS is required. The currently allowed protocol versions for TLS are described in  [[[DK-beveiliging]]] Message service handlers **should NOT** support SSL v3 compatibility mode. |
| Alignment | [Appears as BusinessTransactionCharacteristics/\@isConfidential=transient in CPA.]   | |
| Test References | (empty) |
| Notes  |  For more information see [[[DK-beveiliging]]]  |

### Profile Requirement Item: Persistent Authorization

|[[EBXML-MSG]] Section 4.1.4.7 Persistent Authorization| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Header/eb:Signature`|
| Are persistent authorization methods required? [Yes, for Security Services Profiles 18-21.] Which methods are allowed or required? | **Not applicable**  |
| Alignment | [Appears as BusinessTransactionCharacteristics/\@isAuthorizationRequired=persistent in CPA.]   |  
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Non Persistent Authorization


|[[EBXML-MSG]] Section 4.1.4.8 Non Persistent Authorization | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Header/eb:Signature`|
| Are communication channel authorization methods required? [Yes, for Security Services Profile 2.] Which methods are allowed or required? | TLS client and server authentication is **required** as described in section in 4.2.3. |
| Alignment | [Appears as BusinessTransactionCharacteristics/\@isAuthorizationRequired=transient in CPA.] | |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Trusted Timestamp

|[[EBXML-MSG]] Section 4.1.4.9 Trusted Timestamp | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`/SOAP:Header/eb:Signature`|
| Is a trusted timestamp required? [Yes, for Security Services Profiles 9-12, 15-17, 20, 21.] If so, provide details regarding its usage. | **Not applicable**   |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  |  Applications submitting data to, or receiving data from, Digikoppeling message handlers can perform timestamping. The ebXML Messaging protocol is payload-neutral and therefore supports timestamped payloads. However, this timestamping functionality is not part of the Digikoppeling functionality. Any valid ebXML Message must contain an eb:TimeStamp as part of the eb:MessageData. |

Module : Error Handling
-----------------------

### Profile Requirement Item

|[[EBXML-MSG]] Section 4.2.3.2 Error Element | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|`/SOAP:Header/eb:ErrorList/eb:Error`<br> `/SOAP:Header/eb:ErrorList/ eb:Error/\@codeContext`<br> `/SOAP:Header/eb:ErrorList/ eb:Error/\@errorCode`|
| Is an alternative codeContext used? If so, specify   | **Not applicable**  |
| If an alternative codeContext is used, what is its errorCode list?  |  |
| Profiling (c)   | When errors should be reported to the sending application, how should this be notified (e.g. using a logging mechanism or a proactive callback)?  | **Not applicable**  |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

Module : SyncReply
----------------------

### Profile Requirement Item: SyncReply

|[[EBXML-MSG]] Section 4.3 SyncReply| **Best effort**|**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|---|
|Header elements:<br>`SOAP:Header/eb:SyncReply`||||
| Is SyncReply mode allowed, disallowed, or required, and under what circumstances? [May be process-specific.]  | **Not applicable**.   | SyncReply is restricted to **none** (default) or **mshSignalsOnly** (on condition) Condition for usage of msghSignalsOnly mode is: both parties MSH are able to activate syncReplyMode=msghSignalsOnly see also [Best Practice]| See profile Best Effort or profile Reliable Messaging for details |
| If SyncReply mode is used, are MSH signals, business messages or both expected synchronously?  | **Not applicable** | If SyncReply mode used only MSH signals are expected synchronously  | See profile Reliable Messaging for details   |
| Alignment | [Affects setting of 6.4.7 syncReplyMode element. Appears as MessagingCharacteristics/\@syncReplyMode in CPA.] | |   | |
| Test References | (empty) | |   | |
| Notes  |   | Asynchronous messaging does not preclude support of a “near real time” response quality of service required for e.g. interactive applications. The ebXML MessageId and RefTo MessageId header elements encode correlation of request and response messages. |   | |


Module : Reliable Messaging
---------------------------

### Profile Requirement Item: SOAP Actor attribute

|[[EBXML-MSG]] Section 6.3.1.1 SOAP Actor attribute | **Best effort**|**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|---|
|Header elements:<br>/SOAP:Header/eb:AckRequested/  ||||
| SOAP Actor attribute: Are point-to-point (nextMSH) MSH Acknowledgments to be requested? [Yes, for RM Combinations 1, 3, 5, 7; refer to ebMS section 6.6. Appears as MessagingCharacteristics/\@ackRequested with \@actor=nextMSH in CPA.] | **Not applicable**.  |**Not applicable** | **Not applicable**  |
| Are end-to-end (toParty) MSH Acknowledgments to be requested? [Yes, for RM Combinations 1, 2, 5, 6. Appears as MessagingCharacteristics/\@ackRequested with \@actor=toPartyMSH in CPA.]   | **Not applicable**.  | It is **required** that the final recipient MSH returns a receipt acknowledgment message. | **Optional**: See profiles Best Effort or Reliable Messaging for details. |
| Test References | (empty) |||
| Notes  | (empty) |||

### Profile Requirement Item: Signed attribute

|[[EBXML-MSG]] Section 6.3.1.2 Signed attribute| All profiles:<br> **Best effort**<br>**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|
|Header elements:<br>`/SOAP:Header/eb:AckRequested`/|||
| Must MSH Acknowledgments be (requested to be) signed?  | **Not applicable**.  | **Not applicable**.| Signing of acknowledgements is **required**. |   |
| Alignment | [Appears as MessagingCharacteristics/ \@ackSignatureRequested in CPA.]   |   |
| Test References | (empty) ||
| Notes  | (empty) ||

### Profile Requirement Item: DuplicateElimination

|[[EBXML-MSG]] Section 6.4.1 | **Best effort**|**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|---|
|Header elements:<br>`/SOAP:Header/eb:AckRequested/`  ||||
| Is elimination of duplicate messages required? [Yes, for RM Combinations 1-4.] | Duplicate Elimination is **never used**.  | Duplicate Elimination is **required** | Duplicate Elimination is optional. See profiles Best Effort or Reliable Messaging for details. |
| What is the expected scope in time of duplicate elimination? In other words, how long should messages or message ID's be kept in persistent storage for this purpose? |  (empty) | Message ID's **should** minimally be kept in persistent storage to prevent duplicate delivery during the time interval in which the From Party MSH may be attempting to resend unacknowledged messages. The minimum is (1+Retries)\*RetryInterval.   | (empty)  |
| Alignment | Appears as MessagingCharacteristics/ \@duplicateElimination in CPA |   |   |   |
| Test References | (empty) |  |   |   |
| Notes  |  |   | Message ID's in ebXML are based on [[rfc5322]], and must therefore be globally unique, which in theory prevents accidental re-use of ID's for distinct messages. Factors like system load, disk space, database table limitations, period maintenance schedules **may** be used in message purging policies. Cleaning message ID stores often (temporarily) affects responsiveness of a system. |   |

### Profile Requirement Item: Retries and RetryInterval

|[[EBXML-MSG]]Section 6.4.3, 6.4.4 Retries and RetryInterval | **Best effort**|**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|---|
|Header elements:<br>`/SOAP:Header/eb:AckRequested/` ||||
| (a) If reliable messaging is used, how many times must an MSH attempt to redeliver an unacknowledged message? <br> (b) What is the minimum time a Sending MSH should wait between retries of an unacknowledged message?  | **Not applicable** | Some organizations using the Digikoppeling may not have 24x7 support for their ebXML Messaging services. A system crash may not be remedied until the next working day. Where possible, the values of Retries and RetryInterval **should** be set to allow reliable delivery of messages even after prolonged unavailability. If no value is defined by the parties, a value of 5 days is used. | Depends on the use of best effort or reliable messaging. |
| Alignment | (a) [Appears as ReliableMessaging/Retries in CPA.] (b) [Appears as ReliableMessaging/RetryInterval in CPA.] | | | |
| Test References | (empty) || | |
| Notes  | | If **reliable messaging** is used: Some ebXML messaging software products have a transport retry mechanism, in addition to the ebXML retry mechanism. In this case the ebXML retry interval **should** be set in such a way that any such transport retries have been completed first. | | |

### Profile Requirement Item: PersistDuration

|[[EBXML-MSG]]Section 6.4.6 PersistDuration | **Best effort**|**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|---|
| How long must data from a reliably sent message be kept in persistent storage by a receiving MSH, for the purpose of retransmission? | **Not applicable**   | Depends on the retry interval as defined in the particular collaboration, defined by the involved parties. If no value is defined by the parties, a value of 5 days is used. | Depends on the use of best effort or reliable messaging. |
| Alignment | [Appears as ReliableMessaging/PersistDuration in CPA.] |   |   | |
| Test References | (empty) |||
| Notes  | (empty) |||

### Profile Requirement Item: Reliability Protocol

|[[EBXML-MSG]]Section 6.5.3, 6.5.7 | **Best effort**|**Reliable Messaging**|**End-to-End Security** |
|------------------|---|---|---|
| Usage: required/optional/never used in this profile, Profiled: yes / no   | **Never** used in this profile.   | The Reliable Messaging Protocol in [[EBXML-MSG]] must be used.   | **Optional** in this profile: depends on the use of best effort or reliable messaging. |
| Must a response to a received message be included with the acknowledgment of the received message? Are they to be separate, or are both forms allowed? | **Not applicable**  | Receipt acknowledgment messages are standalone messages. They **must** not to be bundled with business response messages or other ebXML messages. | |
| If a DeliveryFailure error message cannot be delivered successfully, how must the error message's destination party be informed of the problem?  | Each collaborating party is responsible for defining procedures for handling these issues. |   | |
| Alignment | (empty) | ||
| Test References | (empty) |||
| Notes  | (empty) |||

## Module : Message Status

### Profile Requirement Item: Status Request message

|[[EBXML-MSG]] Section 7.1.1 Message Status Request| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`Eb:MessageHeader/eb:StatusRequest`|
| If used, must Message Status Request Messages be digitally signed?   | **Not applicable**. |
| Must unauthorized Message Status Request messages be ignored, rather than responded to, due to security concerns? | **Not applicable**. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: Status Response message

|[[EBXML-MSG]]  Section 7.1.2 Message Status Response| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`Eb:MessageHeader/eb:StatusResponse`|
| If used, must Message Status Response Messages be digitally signed?   | **Not applicable**. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

## Module : Ping Service

### Profile Requirement Item: Ping-Pong Security

|[[EBXML-MSG]] Section 8.1, 8.2 Message Service Handler Ping/Pong Message | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`Eb:MessageHeader/eb:Service`|
| If used, must Ping Messages be digitally signed? | If Ping-Pong is used, it is **optional** for Ping messages to be digitally signed. |
| If used, must Pong Messages be digitally signed? | If Ping-Pong is used, it is **optional** for Pong messages to be digitally signed. |
| Under what circumstances must a Pong Message not be sent? | No recommendation made.   |
| If not supported or unauthorized, must the MSH receiving a Ping respond with an error message, or ignore it due to security concerns?   | No recommendation made |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

## Module : Multi-Hop

### Profile Requirement Item: Use of intermediaries

|[[EBXML-MSG]] Section 10 | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements||
| Are any store-and-forward intermediary MSH nodes present on the message path?  | Endpoints connecting to the Digikoppeling **must** be able to operate in Endpoint mode. They attempt to deliver inbound messages locally, and **may** treat any exceptions as failures. They are **not required** to support any forwarding of ebXML Messages to other business partners.   |
| What are the values of Retry and RetryInterval between intermediate MSH nodes? | **Not applicable**. Any Digikoppeling-level intermediaries must not support reliable messaging, in order to not interfere with end-to-end reliable message delivery. Message handlers **must not** request nextMSH receipt acknowledgments and such requests **should** be ignored by any ebXML intermediary. The ebXML intermediaries also **should not** filter duplicate messages. As with business messages, any Digikoppeling-level ebXML intermediaries **should** attempt to forward end-to-end receipts and errors.   |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  |  In case **Best Effort** is used: Any Digikoppeling-level ebXML intermediary may support transport retries, for instance to handle temporary TCP or HTTP transport level errors. This is not required. In case **Reliable messaging** is used: This profile uses end-to-end reliable messaging. This allows the Digikoppeling to recover from any temporary processing failures at the level of intermediaries. Upcoming versions of the Digikoppeling **may** support store and forward ebXML intermediaries at an infrastructure level. The functionality of these intermediaries is likely be limited to fully transparent, asynchronous store-and-forward routing of ebXML Messages, with the exception of cases as described in par 4.4.1. In the default asynchronous case, no special processing is required of endpoints in the presence of any such intermediaries, as compared to direct point-to-point connections, other than supporting connection to/from the URL and client and server TLS authentication details for the intermediary rather than the “true” sender/recipient. In case **End-to-End Security** is used: see the notes for Best effort of Reliable messaging. |

### Profile Requirement Item: Acknowledgements

|[[EBXML-MSG]] Section 10.1.1, 10.1.3| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header element(s)|`Eb:MessageHeader/`|
| Must each intermediary request acknowledgment from the next MSH? | **Not applicable**. There is **no support** for ebXML next MSH acknowledgments. |
| Must each intermediary return an Intermediate Acknowledgment Message synchronously?   | **Not applicable**. There is **no support** for ebXML next MSH acknowledgments. |
| If both intermediary (multi-hop) and endpoint acknowledgments are requested of the To Party, must they both be sent in the same message? | **Not applicable**. There is **no support** for ebXML next MSH acknowledgments. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

## SOAP Extensions

### Profile Requirement Item: \#wildCard, Id

|[[EBXML-MSG]] Section 2.3.6, 2.3.7, 2.3.8| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| **(Section 2.3.6)** \#wildcard Element Content: Are additional namespace-qualified extension elements required? If so, specify.  | **Not applicable**. No additional namespace-qualified extension elements are required. The toPartyMSH and any intermediaries **must** ignore any extension elements. |
| **(Section 2.3.7)** Is a unique “id” attribute required for each (or any) ebXML SOAP extension element, for the purpose of referencing it alone in a digital signature? | **Not applicable**. Digital Signing is **not supported.**   |
| **(Section 2.3.8)** Is a version other than "2.0" allowed or required for any extension elements?  | These profiles are limited to ebXML Messaging version 2.0 [EBXML-MSG].  |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

## MIME Header Container

### Profile Requirement Item: charset

|[[EBXML-MSG]] Section 2.1.3.2 | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|MIME Header elements|Content-Type|
| Is the "charset" parameter of Content-Type header necessary? If so, what is the (sub)set of allowed values? Example: Content-Type: text/xml; charset="UTF-8" | UTF-8  |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

## HTTP Binding

### Profile Requirement Item: HTTP Headers

|[[EBXML-MSG]] Appendix B.2.2 Sending ebXML Service messages over HTTP| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|MIME parts|
| Is a (non-identity) content-transfer-encoding required for any of the MIME multipart entities? | Content transfer encoding **should not** be used. |
| If other than "ebXML" what must the SOAPAction HTTP header field contain? | The value of the SOAPAction HTTP header field MUST be “ebXML” |
| What additional MIME-like headers must be included among the HTTP headers?   | Additional MIME-like headers **should not** be included with the HTTP header. Any ebXML MSH **should** ignore any such additional HTTP header. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: HTTP Response Codes
|[[EBXML-MSG]] Appendix B.2.3 HTTP Response Codes  | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|MIME parts|
| What client behaviors should result when 3xx, 4xx or 5xx HTTP error codes are received? | In the event of an HTTP 5xx error code, the MSH **must** behave according to the recommendations specified in [[SOAP]]. An HTTP 503 error code **should** be treated as a recoverable error (i.e. **should not** terminate any reliable messaging retries). Codes in the 3xx and 4xx ranges **must** be interpreted as errors. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: HTTP Access Control

|[[EBXML-MSG]] Appendix B.2.6 Access Control Header elements| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|MIME parts|
| Which HTTP access control mechanism(s) are required or allowed? Basic, Digest, or client certificate (the latter only if transport-layer security is used), for example. Refer to item 4.1.4.8 in Security section. | Access control is based on client certificate information only. HTTP Basic or Digest authentication are **not supported**. |
| Alignment | Appears as AccessAuthentication elements in CPA.   |
| Test References | (empty) |
| Notes  | (empty) |

### Profile Requirement Item: HTTP Confidentiality and Security

|[[EBXML-MSG]] Appendix B.2.7 Confidentiality and Transport Protocol Level Security| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|MIME parts |
| Is HTTP transport-layer encryption required? What protocol version(s)? [SSLv3, TLSv1, for example. Refer to item 4.1.4.6 in Security section.] | Encryption is based on HTTPS and TLS. The currently allowed protocol versions for TLS are described in [[[DK-beveiliging]]] Note: TLS implementations **must NOT** support SSL v3 backwards compatiblity mode. |
| What encryption algorithm(s) and minimum key lengths are required?  | The currently allowed protocol versions for TLS are described in [[[DK-beveiliging]]]  |
| What Certificate Authorities are acceptable for server certificate authentication? | PKI overheid maintains a list of approved trusted service providers [[[PKI-CA]]].   |
| Are direct-trust (self-signed) server certificates allowed?   | Self-signed certificates are only allowed in test cases.   |
| Is client-side certificate-based authentication allowed or required?   | Client-side authentication is required.  |
| What client Certificate Authorities are acceptable?  | PKI overheid maintains a list of approved trusted service providers [[[PKI-CA]]].   |
| What certificate verification policies and procedures must be followed?   | PKI overheid procedures are described in [[[PKIO-PvE]]]. The use of certificate revocation lists (CRL) from the trusted CA's is required.  |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  |   For more information see [[[DK-beveiliging]]] |

## SMTP Binding

### Profile Requirement Item: MIME Headers

|[[EBXML-MSG]] Appendix B.3.2 Sending ebXML Messages over SMTP| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Is any specific content-transfer-encoding required, for MIME body parts which must conform to a 7-bit data path? [Base64 or quoted-printable, for example.] | **Not applicable**. This specification only supports the HTTP transport protocol. |
| If other than "ebXML" what must the SOAPAction SMTP header field contain?  | **Not applicable**. This specification only supports the HTTP transport protocol. |
| What additional MIME headers must be included amongst the SMTP headers? | **Not applicable**. This specification only supports the HTTP transport protocol. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |

## Profile Requirement Item: SMTP Confidentiality and Security

|[[EBXML-MSG]] Appendix B.3.4, B.3.5| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
|Header elements|MIME parts |
| What SMTP access control mechanisms are required? [Refer to item 4.1.4.8 in Security section.] | **Not applicable**. This specification only supports the HTTP transport protocol. |
| Is transport-layer security required for SMTP, and what are the specifics of its use? [Refer to item 4.1.4.6 in Security section.] | **Not applicable**. This specification only supports the HTTP transport protocol. |
| Alignment | (empty) |
| Test References | (empty) |
| Notes  | (empty) |
