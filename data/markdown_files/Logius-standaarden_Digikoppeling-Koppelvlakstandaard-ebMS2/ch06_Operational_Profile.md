# Operational Profile

This section defines the operational aspect of the profile: type of deployment
with which the profile which is mentioned above is supposed to operate with,
expected or required conditions of operations, usage context, etc.

## Deployment and Processing requirements for CPAs

|| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Is a specific registry for storing CPA's required? If so, provide details.  | Pending. |
| Is there a set of predefined CPA templates that can be used to create given Parties’ CPA's?   | It is **highly recommended** to use the “CPA Register” facility. A web-based program is available by which CPA's are created and stored. See https://cparegister.minvenj.nl/logius See https://www.logius.nl/diensten/digikoppeling/documentatie for information about the CPA Creation facility (document is written in Dutch). In addition to this there is a **Best Practices** document with information about the use of CPA's. |
| Is there a particular format for file names of CPA's, in case that file name is different from CPA-ID value? | No recommendation.   |
| Others | It is **required** to specify the resulting ebMS collaboration with a CPA. It is **required** that all actions within a CPA make use of (one and) the same default channel for sending acknowledgements. This default channel can only support **one specific profile** within a CPA (for instance either osb-rm-s or osb-rm, not both within one CPA). As a result, when there are actions which are based on different profiles (for instance osb-rm-s and osb-be) and the profiles for the acknowledgements are different as well (for instance osb-rm-s and osb-be), multiple CPA's must be created. |

## Security Profile

|| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Which security profiles are used, and under what circumstances (for which Business Processes)? [Refer to Appendix C of Message Service Specification. May be partially captured by BPSS isConfidential, is Tamperproof, isAuthenticated definitions.] | Security profile 3 [ebMS 2.0] Appendix C]: “Sending MSH authenticates and both MSHs negotiate a secure channel to transmit data” **must be** applied. The HTTPS connection uses encryption to provide in transit confidentiality regarding the complete ebXML message and performs **both** certificate-based Client and Server authentication during the TLS handshake. | Security profile 8 [ebMS 2.0 Appendix C] **must be** used: “Sending MSH applies XML/DSIG structures to message and passes in a secure communications channel. Sending MSH applies XML/DSIG structures to send messagesand Receiving MSH returns a signed receipt.” Security profile 14 [ebMS 2.0 Appendix C] is **optional**: “Sending MSH applies XML/DSIG structures to message **and** applies confidentiality structures (XML-Encryption) and Receiving MSH returns a signed receipt”. |
| (section 4.1.5) Are any recommendations given, with respect to protection or proper handling of MIME headers within an ebXML Message?  | **Not applicable**. No additional recommendations made. |  |   |
| Are any specific third-party security packages approved or required? | No recommendation made.  |  |   |
| Whichsecurity and management policies and practices are recommended? | Pending.  |  |   |
| Any particular procedure for doing HTTP authentication, e.g. if exchanging name and password, how? | Besides the client authentication in HTTPS, no additional procedures are applied. |  |   |
| Others   | (empty) |  

## Reliability Profile

|| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| If reliable messaging is required, by what method(s) may it be implemented? [The ebXML Reliable Messaging protocol, or an alternative reliable messaging or transfer protocol.] | **Not applicable**   | The ebXML reliable messaging protocol **must be** used.  | **Optional**. Depends on the use of best effort or reliable messaging. |
| Which Reliable Messaging feature combinations are required? [Refer to Section 6.6 of Message Service Specification.]   |   | Reliable Messaging profile 2: Duplicate elimination Yes AckRequested ToPartyMSH Yes AckRequested NextMSH No |   |
| Others  |   |

## Error Handling Profile

|[[EBXML-MSG]] Section 4.2.4.2 | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
 (Section 4.2.4.2) Should errors be reported to a URI which is different from the one identified within the From element? What are the requirements for the error reporting URI and the policy for defining it?   | No recommendation made   |
| What is the policy for error reporting? In case an error message cannot be delivered, what other means are used to notify the party, if any?   | Pending.  |
| (Appendix B.4) What communication protocol-level error recovery is required, before deferring to Reliable Messaging recovery? [For example, how many retries should occur in the case of failures in DNS, TCP connection, server errors, timeouts; and at what interval?] | Pending.  |
| Others  |  |

## Message Payload and Flow Profile

| | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| What are typical and maximum message payload sizes which must be handled? (maximum, average) | Some ebXML Messaging products have performance and scalability issues with payloads larger than a (single digit) megabyte in size. Some partners may need to bridge incoming ebXML Message flows to other (enterprise) messaging protocols which have message size limits. Firewalls and other networking equipment may also (implicitly) impose size limits. |
| What are typical communication bandwidth and processing capabilities of an MSH for these Services? | No recommendation made.   |
| Expected Volume of Message flow (throughput): maximum (peak), average?  | No recommendation made.   |
| (Section 2.1.4) How many Payload Containers must be present?   | Messages may contain one or more payload containers  |
| What is the structure and content of each container? [List MIME Content-Types and other process-specific requirements.] Are there restrictions on the MIME types allowed for attachments? | Each payload container will get a MIME type reflecting the type of the ‘content’ it contains.  |
| How is each container distinguished from the others? [By a fixed ordering of containers, a fixed Manifest ordering, or specific Content-ID values.]. Any expected relative order of attachments of various types? | No recommendation made.   |
| Others   |   |

## Additional Messaging Features beyond ebMS Specification

| | All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Are there additional features out of specification scope, whichare part of this messaging profile, as an extension to the ebMS profiling? | No. |

## Additional Deployment or Operational Requirements

|| All profiles:<br> **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** |
|------------------|---|
| Operational or deployment aspects which are object to further requirements or recommendations. | Pending.  |
