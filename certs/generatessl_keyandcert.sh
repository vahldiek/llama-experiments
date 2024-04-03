#!/usr/bin/env bash
# check if command line argument is empty or not present
if [ "$1" == "" ] || [ $# -gt 1 ]; then
        echo "Usage $0 <IP Address>"
        exit 1
fi

export IP_ADDR_PLACEHOLDER=$1
envsubst '$IP_ADDR_PLACEHOLDER' < streamlit_csr.tpl > streamlit_csr.cnf
envsubst '$IP_ADDR_PLACEHOLDER' < streamlit.tpl > streamlit.cnf

#generate a new TLS/SSL key
openssl genrsa -out streamlit.key 2048

#generate a new certificate signing request
openssl req -new -key streamlit.key -out streamlit.csr -config streamlit_csr.cnf

#ask the local CA to generate and sign the new certificate
openssl x509 -req -in streamlit.csr -CA demoRootCA.crt -CAkey demoRootCA.key -CAcreateserial -out streamlit.crt -days 730 -sha256 -extfile streamlit.cnf

#dump the contents of the new cert
openssl x509 -in streamlit.crt -noout -text
