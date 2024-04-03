basicConstraints       = CA:FALSE
authorityKeyIdentifier = keyid:always, issuer:always
keyUsage               = nonRepudiation, digitalSignature, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName         = @alt_names

[ alt_names ]
IP.1 = $IP_ADDR_PLACEHOLDER
