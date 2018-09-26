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
            warnings ([canComputeNew: false, canResolveRelativePaths: false, canRunOnFailed: true,
                    categoriesPattern: '', defaultEncoding: '', excludePattern: '',
                    failedTotalAll: '0', healthy: '', includePattern: '', messagesPattern: '',
                    parserConfigurations: [
                       // [parserName: 'PyLintPio', pattern: 'pylint.log'],
                       // [parserName: 'MyPy', pattern: 'mypy.log']
                    ], unHealthy: ''])
            publishHTML([allowMissing: true, alwaysLinkToLastBuild: false, keepAll: true,
                    reportDir: 'cover_html', reportFiles: 'index.html',
                    reportName:  'Coverage Report', reportTitles: 'Coverage Report'])
         }
    }
}