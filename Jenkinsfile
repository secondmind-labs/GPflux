pipeline {

    agent {
        label 'linux'
    }

    stages {
        stage('Test'){
            agent {
                dockerfile {
                    filename 'Dockerfile'
                    registryUrl 'https://eu.gcr.io/prowlerio-docker'
                    registryCredentialsId 'gcr:prowlerio-docker'
                }
            }
            steps {
                sh "git clone --depth 1 --branch develop https://github.com/GPflow/GPflow.git"
                sh "pip install tox==3.2.1"
                sh "tox -e jenkins"

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
            post {
                always {
                    junit '**/reports/*.xml'
                    cleanWs notFailBuild: true
                }
            }
        }
    }
}
