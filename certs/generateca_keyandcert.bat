echo You will be prompted to create a password to protect the new CA key
openssl genrsa -des3 -out demoRootCA.key 2048
echo You will be prompted to use the password that you just created
openssl req -new -nodes -x509 -days 730 -key demoRootCA.key -out demoRootCA.crt -config ./demoRootCA.cnf
