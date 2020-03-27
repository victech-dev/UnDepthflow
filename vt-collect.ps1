Write-Host "Collecting from VT-Learner-00 ..." -ForegroundColor Green		
Remove-Item ./.results00 -Force -Recurse -ErrorAction SilentlyContinue
mkdir .results00 | Out-Null
plink -load VT-Learner-00 -batch "cd /workspace/UnDepthflow; tar -cvf results.tar .results_*"
pscp -load VT-Learner-00 -batch -r dev@192.168.2.50:/workspace/UnDepthflow/results.tar C:/workspace/UnDepthflow/.results00/results.tar
tar -xvf .results00/results.tar -C .results00
del .results00/results.tar
plink -load VT-Learner-00 -batch "rm -f /workspace/UnDepthflow/results.tar"

Write-Host "Collecting from VT-Learner-01 ..." -ForegroundColor Green		
Remove-Item ./.results01 -Force -Recurse -ErrorAction SilentlyContinue
mkdir .results01 | Out-Null
plink -load VT-Learner-01 -batch "cd /workspace/UnDepthflow; tar -cvf results.tar .results_*"
pscp -load VT-Learner-01 -batch -r dev@192.168.2.51:/workspace/UnDepthflow/results.tar C:/workspace/UnDepthflow/.results01/results.tar
tar -xvf .results01/results.tar -C .results01
del .results01/results.tar
plink -load VT-Learner-01 -batch "rm -f /workspace/UnDepthflow/results.tar"

Write-Host "Collecting from VT-Learner-02 ..." -ForegroundColor Green		
Remove-Item ./.results02 -Force -Recurse -ErrorAction SilentlyContinue
mkdir .results02 | Out-Null
plink -load VT-Learner-02 -batch "cd /workspace/UnDepthflow; tar -cvf results.tar .results_*"
pscp -load VT-Learner-02 -batch -r dev@192.168.2.52:/workspace/UnDepthflow/results.tar C:/workspace/UnDepthflow/.results02/results.tar
tar -xvf .results02/results.tar -C .results02
del .results02/results.tar
plink -load VT-Learner-02 -batch "rm -f /workspace/UnDepthflow/results.tar"

Write-Host "Collecting from VT-Learner-99 ..." -ForegroundColor Green		
Remove-Item ./.results99 -Force -Recurse -ErrorAction SilentlyContinue
mkdir .results99 | Out-Null
plink -load VT-Learner-99 -batch "cd /workspace/UnDepthflow; tar -cvf results.tar .results_*"
pscp -load VT-Learner-99 -batch -r ubuntu@14.49.44.113:/workspace/UnDepthflow/results.tar C:/workspace/UnDepthflow/.results99/results.tar
tar -xvf .results99/results.tar -C .results99
del .results99/results.tar
plink -load VT-Learner-99 -batch "rm -f /workspace/UnDepthflow/results.tar"

pause