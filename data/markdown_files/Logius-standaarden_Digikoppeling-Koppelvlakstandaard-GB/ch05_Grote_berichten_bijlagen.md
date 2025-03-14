# Grote berichten bijlagen

## XSD voor DK GB PULL principe

```XML
<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
  elementFormDefault="qualified"
  xmlns:tns="http://www.logius.nl/digikoppeling/gb/2010/10"
  targetNamespace="http://www.logius.nl/digikoppeling/gb/2010/10">
 <xs:element name="digikoppeling-external-data-references" type="tns:external-data-reference">
  </xs:element>
  
  <xs:complexType name="external-data-reference">
    <xs:sequence>
      <xs:element name="data-reference" maxOccurs="unbounded"
                     type="tns:data-reference" />
    </xs:sequence>
    <xs:attribute name="profile" type="tns:gb-profile" />
  </xs:complexType>


    <xs:complexType name="data-reference">
      <xs:sequence minOccurs="1">
        <xs:element name="lifetime">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="creationTime" type="tns:datetimeType"   
                                minOccurs="0"  />
              <xs:element name="expirationTime" type="tns:datetimeType"   
                                minOccurs="0" />
            </xs:sequence>
          </xs:complexType>
        </xs:element>
        <xs:element name="content">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="filename" type="xs:NCName" />
              <xs:element name="checksum" type="tns:checksumType" />
              <xs:element name="size" type="xs:unsignedLong" />
            </xs:sequence>
            <xs:attribute name="contentType" use="required" type="xs:string"/>
          </xs:complexType>
        </xs:element>
        <xs:element name="transport">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="location">
                <xs:complexType>
                  <xs:choice>
                  <xs:element name="senderUrl" type="tns:urlType" />
                  <xs:element name="receiverUrl" type="tns:urlType" />
                  </xs:choice>
                </xs:complexType>
              </xs:element>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
      </xs:sequence>
      <xs:attribute name="contextId" use="optional"/>
    </xs:complexType>


  <xs:simpleType name="gb-profile" final="restriction">
   <xs:restriction base="xs:string">
      <xs:enumeration value="digikoppeling-gb-1.0" />
      <!--
        DigiKoppeling GB profiel 1 aanduiding
      -->
    </xs:restriction>
  </xs:simpleType>

 <xs:complexType name="datetimeType">
    <xs:simpleContent>
      <xs:extension base="xs:dateTime">
        <xs:attribute name="type" use="required"
                      type="xs:string" fixed="xs:dateTime" />
      </xs:extension>
    </xs:simpleContent>
  </xs:complexType>

 <xs:complexType name="checksumType">
   <xs:simpleContent>
      <xs:extension base="tns:md5String">
         <xs:attribute name="type" use="required">
           <xs:simpleType>
            <xs:restriction base="xs:string">
              <xs:enumeration value="MD5" />
            <xs:enumeration value="SHA1" />
            <xs:enumeration value="SHA256" />
            <xs:enumeration value="SHA384" />
            <xs:enumeration value="SHA512" />          
            </xs:restriction>
           </xs:simpleType>
        </xs:attribute>
      </xs:extension>
    </xs:simpleContent>
 </xs:complexType>
    
 <xs:complexType name="urlType">
   <xs:simpleContent>
      <xs:extension base="tns:anyString">
       <xs:attribute name="type" use="required" fixed="xs:anyURI" />
     </xs:extension>
    </xs:simpleContent>
  </xs:complexType>

 <xs:complexType name="md5String">
   <xs:simpleContent>
      <xs:restriction base="tns:anyString">
       <xs:pattern value="[0-9a-fA-F]*" />            
      </xs:restriction>
    </xs:simpleContent>
  </xs:complexType>

 <xs:complexType name="anyString">
   <xs:simpleContent>
      <xs:extension base="xs:string" />              
    </xs:simpleContent>
  </xs:complexType>

</xs:schema>
```

## XML Voorbeeld PULL Bericht

Dit hoofdstuk presenteert een voorbeeld van de metadata van een bestand bij gebruik PULL principe

```XML
<?xml version="1.0" encoding="UTF-8"?>
<tns:digikoppeling-external-data-references
  profile="digikoppeling-gb-1.0"
  xmlns:tns="http://www.logius.nl/digikoppeling/gb/2010/10"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.logius.nl/digikoppeling/gb/2010/10/gb-meta.xsd"> 
  <tns:data-reference contextId="12345">
    <tns:lifetime>
      <tns:creationTime type="xs:dateTime">2001-12-31T12:00:00Z</tns:creationTime>
      <tns:expirationTime type="xs:dateTime">2001-12-31T12:00:00Z</tns:expirationTime>
    </tns:lifetime>
    <tns:content contentType="application/xml">
      <tns:filename>NCName</tns:filename>
      <tns:checksum type="MD5">0123456789abcdef0123456789abcdef</tns:checksum>
      <tns:size>0</tns:size>
    </tns:content>
    <tns:transport>
      <tns:location>
        <tns:senderUrl type="xs:anyURI">https://any.url/any.name</tns:senderUrl>
      </tns:location>
    </tns:transport>
  </tns:data-reference>
</tns:digikoppeling-external-data-references>
```


## XSD voor DK GB PUSH principe

```XML
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" elementFormDefault="qualified" xmlns:tns="http://www.logius.nl/digikoppeling/gb/2020/09" targetNamespace="http://www.logius.nl/digikoppeling/gb/2020/09">
  <xs:element name="digikoppeling-external-data-references-request" type="tns:external-data-reference-request" />
  <xs:complexType name="external-data-reference-request">
    <xs:sequence>
      <xs:element name="data-reference-request" maxOccurs="unbounded" type="tns:data-reference-request" />
    </xs:sequence>
    <xs:attribute name="profile" type="tns:gb-profile" />
  </xs:complexType>
  <xs:complexType name="data-reference-request">
    <xs:sequence minOccurs="1">
      <xs:element name="compression" type="tns:compression" />
      <xs:element name="content">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="filename" type="xs:string" />
            <xs:element name="checksum" type="tns:checksumType" />
            <xs:element name="size" type="xs:unsignedLong" />
            <xs:element name="transport">
              <xs:complexType>
                <xs:sequence>
                  <xs:element name="location" type="tns:location" />
                  <xs:element name="part" minOccurs="0" maxOccurs="unbounded">
                    <xs:complexType>
                      <xs:sequence>
                        <xs:element name="filename" type="xs:string" />
                        <xs:element name="checksum" type="tns:checksumType" />
                        <xs:element name="size" type="xs:unsignedLong" />
                      </xs:sequence>
                    </xs:complexType>
                  </xs:element>
                </xs:sequence>
              </xs:complexType>
            </xs:element>
          </xs:sequence>
          <xs:attribute name="contentType" use="required" type="xs:string" />
        </xs:complexType>
      </xs:element>
    </xs:sequence>
    <xs:attribute name="contextId" use="optional" />
  </xs:complexType>
  <xs:element name="digikoppeling-external-data-references-response" type="tns:external-data-reference-response" />
  <xs:complexType name="external-data-reference-response">
    <xs:sequence>
      <xs:element name="data-reference-response" maxOccurs="unbounded" type="tns:data-reference-response" />
    </xs:sequence>
    <xs:attribute name="profile" type="tns:gb-profile" />
  </xs:complexType>
  <xs:complexType name="data-reference-response">
    <xs:sequence minOccurs="1">
      <xs:element name="compression" type="tns:compression" />
      <xs:element name="content">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="filename" type="xs:string" />
            <xs:element name="checksum" type="tns:checksumType" />
            <xs:element name="size" type="xs:unsignedLong" />
            <xs:element name="status" type="tns:status" />
            <xs:element name="reason" type="xs:string" minOccurs="0" />
            <xs:element name="transport">
              <xs:complexType>
                <xs:sequence>
                  <xs:element name="location" type="tns:location" />
                  <xs:element name="part" minOccurs="0" maxOccurs="unbounded">
                    <xs:complexType>
                      <xs:sequence>
                        <xs:element name="filename" type="xs:string" />
                        <xs:element name="checksum" type="tns:checksumType" />
                        <xs:element name="size" type="xs:unsignedLong" />
                        <xs:element name="status" type="tns:status" />
                        <xs:element name="reason" type="xs:string" minOccurs="0" />
                      </xs:sequence>
                    </xs:complexType>
                  </xs:element>
                </xs:sequence>
              </xs:complexType>
            </xs:element>
          </xs:sequence>
          <xs:attribute name="contentType" use="required" type="xs:string" />
        </xs:complexType>
      </xs:element>
    </xs:sequence>
    <xs:attribute name="contextId" use="optional" />
  </xs:complexType>
  <xs:complexType name="location">
    <xs:choice>
       <xs:element name="receiverUrl" type="tns:urlType" />
    </xs:choice>
  </xs:complexType>
  <xs:simpleType name="gb-profile" final="restriction">
    <xs:restriction base="xs:string">
      <xs:enumeration value="digikoppeling-gb-4.0" />
      <!-- DigiKoppeling GB profiel aanduiding -->
    </xs:restriction>
  </xs:simpleType>
  <xs:complexType name="datetimeType">
    <xs:simpleContent>
      <xs:extension base="xs:dateTime">
        <xs:attribute name="type" use="required" type="xs:string" fixed="xs:dateTime" />
      </xs:extension>
    </xs:simpleContent>
  </xs:complexType>
  <xs:complexType name="checksumType">
    <xs:simpleContent>
      <xs:extension base="tns:checksumString">
        <xs:attribute name="type" use="required">
          <xs:simpleType>
            <xs:restriction base="xs:string">
              <xs:enumeration value="MD5" />
              <xs:enumeration value="SHA1" />
              <xs:enumeration value="SHA256" />
              <xs:enumeration value="SHA384" />
              <xs:enumeration value="SHA512" />
            </xs:restriction>
          </xs:simpleType>
        </xs:attribute>
      </xs:extension>
    </xs:simpleContent>
  </xs:complexType>
  <xs:simpleType name="compression">
    <xs:restriction base="xs:string">
      <xs:enumeration value="NONE" />
      <xs:enumeration value="ZIP4J" />
    </xs:restriction>
  </xs:simpleType>
  <xs:complexType name="urlType">
    <xs:simpleContent>
      <xs:extension base="tns:anyString">
        <xs:attribute name="type" use="required" fixed="xs:anyURI" />
      </xs:extension>
    </xs:simpleContent>
  </xs:complexType>
  <xs:complexType name="checksumString">
    <xs:simpleContent>
      <xs:restriction base="tns:anyString">
        <xs:pattern value="[0-9a-fA-F]*" />
      </xs:restriction>
    </xs:simpleContent>
  </xs:complexType>
  <xs:simpleType name="status">
    <xs:restriction base="xs:string">
      <xs:enumeration value="OK" />
      <xs:enumeration value="FILE_NOT_FOUND" />
      <xs:enumeration value="CHECKSUM_TYPE_NOT_SUPPORTED" />
      <xs:enumeration value="CHECKSUM_ERROR" />
      <xs:enumeration value="INCORRECT_FILE_SIZE" />
      <xs:enumeration value="COMPRESSION_NOT_SUPPORTED" />
      <xs:enumeration value="DECOMPRESSION_ERROR" />
      <xs:enumeration value="UNKNOWN_ERROR" />
    </xs:restriction>
  </xs:simpleType>
  <xs:complexType name="anyString">
    <xs:simpleContent>
      <xs:extension base="xs:string" />
    </xs:simpleContent>
  </xs:complexType>
</xs:schema>
```



## XML Voorbeeld PUSH Bericht

### Data-reference-request bericht 1 (PUSH)

Hieronder volgt een voorbeeld van een Grote Berichten data-reference-request bericht voor een PDF bestand genaamd file.pdf met een grootte van 2048MB, die is ge-upload:

```XML
<gb:digikoppeling-external-data-references-request profile="digikoppeling-gb-4.0"
  xmlns:gb="http://www.logius.nl/digikoppeling/gb/2020/09" 
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" >
   <gb:data-reference-request>
    <gb:compression>NONE</gb:compression>
    <gb:content contentType="application/pdf">
      <gb:filename>file.pdf</gb:filename>
      <gb:checksum type="MD5">01234567890123456789012345678901</gb:checksum>
      <gb:size>2048</gb:size>
      <gb:transport>
        <gb:location>
          <gb:receiverUrl type="xs:anyURI">https://my.host.nl/files/file.pdf</gb:receiverUrl>
        </gb:location>
      </gb:transport>
    </gb:content>
  </gb:data-reference-request>
</gb:digikoppeling-external-data-references-request>
```

### Data-reference-response bericht 1 (PUSH)

Hieronder volgt een voorbeeld van een Grote Berichten data-reference-response bericht voor de PDF bestand genaamd file.pdf met een grootte van 2048MiB, die is ge-upload en gezipped 
Waarbij file.pdf niet is gevonden.

```XML
<gb:digikoppeling-external-data-references-response profile="digikoppeling-gb-4.0"
  xmlns:gb="http://www.logius.nl/digikoppeling/gb/2020/09" 
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" >                                            
  <gb:data-reference-response>
    <gb:compression>NONE</gb:compression>
    <gb:content contentType="application/pdf">
      <gb:filename>file.pdf</gb:filename>
      <gb:checksum type="MD5">01234567890123456789012345678901</gb:checksum>
      <gb:size>2048</gb:size>
      <gb:status>FILE_NOT_FOUND</gb:status>
      <gb:transport>
        <gb:location>
          <gb:receiverUrl type="xs:anyURI">https://my.host.nl/files/file.pdf</gb:receiverUrl>
        </gb:location>
      </gb:transport>
    </gb:content>
  </gb:data-reference-response>
</gb:digikoppeling-external-data-references-response>
```

Alle errors behalve UNKNOWN_ERROR zijn recoverable en hebben geen reason nodig.

### Data-reference-request bericht 2 (PUSH)

Hieronder volgt een voorbeeld van een Grote Berichten data-reference-request bericht voor een PDF bestand genaamd `file.pdf` met een grootte van 2048MiB, die moet worden ge-upload en is gezipped in de volgende 2 zip parts:

- file.pdf.z01 met een grootte van 1024MiB is ge-upload naar `https://my.host.nl/files/file.pdf.z01`
- file.pdf.zip met een grootte van 765MiB is ge-upload naar `https://my.host.nl/files/file.pdf.zip`

```XML
<gb:digikoppeling-external-data-references-request profile="digikoppeling-gb-4.0"                                                 
  xmlns:gb="http://www.logius.nl/digikoppeling/gb/2020/09" 
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <gb:data-reference-request>
    <gb:compression>ZIP4J</gb:compression>
    <gb:content contentType="application/pdf">
      <gb:filename>file.pdf</gb:filename>
      <gb:checksum type="MD5">01234567890123456789012345678901</gb:checksum>
      <gb:size>2048</gb:size>
      <gb:transport>
        <gb:location>
          <gb:receiverUrl type="xs:anyURI">https://my.host.nl/files/</gb:receiverUrl>
        </gb:location>
        <gb:part>
          <gb:filename>file.pdf.z01</gb:filename>
          <gb:checksum type="MD5">12345678901234567890123456789012</gb:checksum>
          <gb:size>1024</gb:size>
        </gb:part>
        <gb:part>
          <gb:filename>file.pdf.zip</gb:filename>
          <gb:checksum type="MD5">23456789012345678901234567890123</gb:checksum>
          <gb:size>765</gb:size>
        </gb:part>
      </gb:transport>
    </gb:content>
  </gb:data-reference-request>
</gb:digikoppeling-external-data-references-request>
```

### Data-reference-response bericht 2 (PUSH)

Hieronder volgt een voorbeeld van een Grote Berichten data-reference-response bericht voor de PDF bestand genaamd file.pdf met een grootte van 2048MB, die is ge-upload en gezipped in de volgende 2 zip parts:

- file.001.zip met een grootte van 1024MiB is ge-upload naar `https://my.host.nl/files/file.001.zip`
- file.002.zip met een grootte van 765MiB is ge-upload naar `https://my.host.nl/files/file.002.zip`

Waarbij `file.001.zip` correct is geupload en `file.002.zip` niet is gevonden.

```XML
<gb:digikoppeling-external-data-references-response profile="digikoppeling-gb-4.0"                                                 
  xmlns:gb="http://www.logius.nl/digikoppeling/gb/2020/09" 
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" >
  <gb:data-reference-response>
    <gb:compression>ZIP4J</gb:compression>
    <gb:content contentType="application/pdf">
      <gb:filename>file.pdf</gb:filename>
      <gb:checksum type="MD5">01234567890123456789012345678901</gb:checksum>
      <gb:size>2048</gb:size>
      <gb:status>FILE_NOT_FOUND</gb:status>
      <gb:reason></gb:reason>
      <gb:transport>
        <gb:location>
          <gb:receiverUrl type="xs:anyURI">https://my.host.nl/files/</gb:receiverUrl>
        </gb:location>
        <gb:part>
          <gb:filename>file.pdf.z01</gb:filename>
          <gb:checksum type="MD5">12345678901234567890123456789012</gb:checksum>
          <gb:size>1024</gb:size>
          <gb:status>OK</gb:status>
        </gb:part>
        <gb:part>
          <gb:filename>file.pdf.zip</gb:filename>
          <gb:checksum type="MD5">23456789012345678901234567890123</gb:checksum>
          <gb:size>765</gb:size>
          <gb:status>FILE_NOT_FOUND</gb:status>
        </gb:part>
      </gb:transport>
    </gb:content>
  </gb:data-reference-response>
</gb:digikoppeling-external-data-references-response>
```

Alle errors behalve UNKNOWN_ERROR zijn recoverable en hebben geen reason nodig.

