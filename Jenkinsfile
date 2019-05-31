pipeline {

    agent {
        label 'linux'
    }

    stages {
        stage('Test'){
            agent {
                docker {
                    image 'python:3.7-stretch'
                    args '-v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/ca-certificates.crt --user root'
                }
            }
            steps {
                sh "pip install tox==3.2.1"
                sh "tox"

                publishHTML([
                    allowMissing: true,
                    alwaysLinkToLastBuild: false,
                    keepAll: true,
                    reportDir: 'cover_html',
                    reportFiles: 'index.html',
                    reportName:  'Coverage Report',
                    reportTitles: 'Coverage Report'
                ])
            }
        }
        stage('Upload Development Release'){
            when {
                branch 'release/**'
            }
            steps {
                 withCredentials([
                   string(credentialsId: 'DevpiProwlerioPassword', variable: 'PIO_PASSWORD'),
                ]) {
                    sh "devpi use https://devpi.piointernal.prowler.io/prowler-io/dev"
                    sh "devpi login prowler-io --password=${PIO_PASSWORD}"
                    sh "devpi upload"
                }
            }
        }
        stage('Upload Production Release'){
            when {
                branch 'master'
            }
            steps {
                 withCredentials([
                   string(credentialsId: 'DevpiProwlerioPassword', variable: 'PIO_PASSWORD'),
                ]) {
                    sh "devpi use https://devpi.piointernal.prowler.io/prowler-io/prod"
                    sh "devpi login prowler-io --password=${PIO_PASSWORD}"
                    sh "devpi upload"
                }
            }
        }
    }
}
