@echo off
if [%1]==[] goto usage


REM replace IP address placeholder with real IP address
copy streamlit_csr.tpl streamlit_csr.cnf
copy streamlit.tpl streamlit.cnf
cscript replace.vbs streamlit_csr.cnf "IP_ADDR_PLACEHOLDER" %1
cscript replace.vbs streamlit.cnf "IP_ADDR_PLACEHOLDER" %1

REM generate a new TLS/SSL key
openssl genrsa -out streamlit.key 2048

REM generate a new certificate signing request
openssl req -new -key streamlit.key -out streamlit.csr -config streamlit_csr.cnf

REM ask the local CA to generate and sign the new certificate
openssl x509 -req -in streamlit.csr -CA demoRootCA.crt -CAkey demoRootCA.key -CAcreateserial -out streamlit.crt -days 730 -sha256 -extfile streamlit.cnf

REM dump the contents of the new cert
openssl x509 -in streamlit.crt -noout -text

goto :eof

:usage
@echo Usage: %0 ^<IP address^>

