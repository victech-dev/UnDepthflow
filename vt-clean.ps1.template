# Clean all results directories of each learning machine

$cmd = @"
cd /workspace/UnDepthflow;
rm -rf .results*
"@

$confirmation = Read-Host "Are you Sure You Want To Clean All (say yes)"
if ($confirmation -eq 'yes') {
	Write-Host "Cleaning VT-Learner-00 ..." -ForegroundColor Green		
	plink -load VT-Learner-00 -batch "$cmd"
	Write-Host "Cleaning VT-Learner-01 ..." -ForegroundColor Green		
	plink -load VT-Learner-01 -batch "$cmd"
	Write-Host "Cleaning VT-Learner-02 ..." -ForegroundColor Green		
	plink -load VT-Learner-02 -batch "$cmd"
	Write-Host "Cleaning VT-Learner-99 ..." -ForegroundColor Green		
	plink -load VT-Learner-99 -batch "$cmd"
	Write-Host "Done ..." -ForegroundColor Green		
	pause
}

