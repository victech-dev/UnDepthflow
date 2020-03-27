$cmd00 = @"
cd /workspace/UnDepthflow;
git pull;
tmux kill-session -t sehee &> /dev/null;
tmux new-session -d -s sehee;
tmux send-keys -t sehee:0 'python3 main.py --smooth_mode=undepthflow_v2 --depth_smooth_weight=0.5 2> tlog.txt' Enter;
"@

$cmd01 = @"
cd /workspace/UnDepthflow;
git pull;
tmux kill-session -t sehee &> /dev/null;
tmux new-session -d -s sehee;
tmux send-keys -t sehee:0 'python3 main.py --smooth_mode=undepthflow_v2 --depth_smooth_weight=1.0 2> tlog.txt' Enter;
"@

$cmd02 = @"
cd /workspace/UnDepthflow;
git pull;
tmux kill-session -t sehee &> /dev/null;
tmux new-session -d -s sehee;
tmux send-keys -t sehee:0 'python3 main.py --smooth_mode=undepthflow_v2 --depth_smooth_weight=1.5 2> tlog.txt' Enter;
"@

$cmd99 = @"
cd /workspace/UnDepthflow;
git pull;
tmux kill-session -t sehee &> /dev/null;
tmux new-session -d -s sehee;
tmux send-keys -t sehee:0 '. activate vt-learn' Enter;
tmux send-keys -t sehee:0 'python3 main.py --smooth_mode=undepthflow_v2 --depth_smooth_weight=2.0 2> tlog.txt' Enter;
"@


Write-Host "Sending command to VT-Learner-00 ..." -ForegroundColor Green		
plink -load VT-Learner-00 -batch "$cmd00"
Write-Host "Sending command to VT-Learner-01 ..." -ForegroundColor Green		
plink -load VT-Learner-01 -batch "$cmd01"
Write-Host "Sending command to VT-Learner-02 ..." -ForegroundColor Green		
plink -load VT-Learner-02 -batch "$cmd02"
Write-Host "Sending command to VT-Learner-99 ..." -ForegroundColor Green		
plink -load VT-Learner-99 -batch "$cmd99"
Write-Host "Done ..." -ForegroundColor Green		
pause