pipeline {

    agent {
        label 'linux'
    }

    stages {
        stage('Test'){
            steps {
                sh "tox"
            }
        }
    }
    post {
         always {
            // Report results
            junit '**/nosetests.xml'

            publishHTML([allowMissing: true, alwaysLinkToLastBuild: false, keepAll: true,
                    reportDir: 'cover_html', reportFiles: 'index.html',
                    reportName:  'Coverage Report', reportTitles: 'Coverage Report'])
         }
    }
}