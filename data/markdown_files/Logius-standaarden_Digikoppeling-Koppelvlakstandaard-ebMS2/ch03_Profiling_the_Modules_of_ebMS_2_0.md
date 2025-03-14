# Profiling the Modules of ebMS 2.0

## Core Modules

### Core Extension Elements [ebMS 2.0] Section 3 

|Profile(s)             | Usage: required/optional/never used in this profile |
|-------------------|---|
| **Best effort**<br>**Reliable Messaging**<br>**End-to-End Security** | Support is **required**. |

### Security Module [ebMS 2.0] Section 4.1

| Profile(s)| Usage: required/optional/never used in this profile |
|------------------|---|
| **Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security** | The Security Module is **required** in this profile. <br>Security profile 3 [[EBXML-MSG]]/Appendix C **must be used**: “Sending MSH authenticates and both MSH's negotiate a secure channel to transmit data”. The HTTPS connection uses encryption to provide in transit confidentiality of the complete ebXML message and performs both certificate-based Client and Server authentication during the TLS handshake.  |
|**End-to-End Security** |Security profile 8 [[EBXML-MSG]]/Appendix C **must be used**: “Sending MSH applies XML/DSIG structures to message and passes in secure communications channel. Sending MSH applies XML/DSIG structures to message and Receiving MSH returns a signed receipt.”|

### SyncReply Module [ebMS 2.0] Section 4.3

| Profile(s)| Usage: required/optional/never used in this profile |Notes|
|------------------|---|---|
|**Best effort**|**Never used** in this profile|(empty)|
|**Reliable Messaging**|**Optional used** in this profile. All messages, including acknowledgments and error messages, are sent asynchronously, with the exception of cases as described in par 4.4.1. Only in specific cases can MSH signals (acknowledgements, errors) sent synchronously. See 4.4.1 for conditions.|Asynchronous messaging does not preclude fast response times, as is required to support interactive applications. Asynchronous messaging supports higher levels of scalability and supports scenarios where a response message may be sent minutes, hours or days after the initial request message. Asynchronous messaging may be combined transparently with store-and-forward intermediary|
|**End-to-End Security**|**Optional used** in this profile. See profile Best Effort or profile Reliable Messaging for details|(empty)|


## Additional Modules

### Reliable Messaging Module [ebMS 2.0] Section 6

| Profile(s)| Usage: required/optional/never used in this profile |Notes|
|------------------|---|---|
|**Best effort**|**Never used** in this profile. |The ebXML reliable messaging protocol is not used. Acknowledgment Messages must not be sent or requested, and the receiver should not eliminate duplicate messages. |
|**Reliable Messaging**|**Required** in this profile. Reliable Messaging profile 2, Once-And-Only-Once Reliable Messaging at the End-To-End level only based upon end-to-end retransmission. |In this profile the FromParty MSH (message origination) must request, and the ToParty MSH (message final destination) must send an acknowledgment message. The ToParty MSH must also filter any duplicate messages based on ebXML MessageId. Any intermediate NextMSH ebXML-aware nodes (see caveat in section 'Multi-Hop Module' in this chapter) have no reliable messaging functionality. Acknowledgment messages must not be consumed by any such intermediary but routed like any ebXML Message back to the original (true) sender. |
|**End-to-End Security**|**Optional used** in this profile. See profile Best Effort or profile Reliable Messaging for details. |(empty)|

### Message Status Service [ebMS 2.0] Section 7

| Profile(s)| Usage: required/optional/never used in this profile |Notes|
|------------------|---|---|
|**Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security**|**Optional**. Message Status Service is not required in these profiles. |(empty) |


### Ping Service [ebMS 2.0] Section 8

| Profile(s)| Usage: required/optional/never used in this profile |Notes|
|------------------|---|---|
|**Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security**|Ping Service is **not required** in these profiles.|(empty) |

### Message Order [ebMS 2.0] Section 9 

| Profile(s)| Usage: required/optional/never used in this profile |Notes|
|------------------|---|---|
|**Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security**|**Optional**. Message Order is *strongly discouraged* in these profiles.|Many organisations use message handlers that do not support this functionality. Therefore, it can only be used if communicating parties agree to this option in advance. This specification is limited to message service handler order functionality and does not preclude application-level in-order processing if sequence information is somehow provided at the business document level.|

### Multi-Hop Module [ebMS 2.0] Section 10

| Profile(s)| Usage: required/optional/never used in this profile |Notes|
|------------------|---|---|
|**Best effort**,<br>**Reliable Messaging**,<br>**End-to-End Security**|**Never used** in this profile.|Multi-hop is the process of passing the message through one or more intermediary nodes or MSH's. An Intermediary is any node or MSH where the message is received, but is not the Sending or Receiving MSH endpoint. This node is called an Intermediary.|

