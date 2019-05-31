FROM python:3.7-stretch

RUN apt-get update && apt-get install git ca-certificates
#RUN apk add --no-cache git ca-certificates

# Configure pip to use the Prowler.io root certificate
ENV PIP_CERT=/usr/local/share/ca-certificates/prowlerio_ca.crt

# Install PROWLER.io's CA certificate so that pip can use devpi.piointernal.prowler.io
ADD https://downloads.piointernal.prowler.io/prowlerio/ca-certificates/prowlerio_ca.crt $PIP_CERT
RUN update-ca-certificates

# Configure pip to use devpi.piointernal.prowler.io
ENV PIP_INDEX_URL=https://devpi.piointernal.prowler.io/prowler-io/prod/+simple/

RUN  pip install tox==3.2.1

ADD . /src

WORKDIR /src
