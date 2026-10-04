@echo off
echo ========================================================
echo Starting DUSt3R 3D Vision Demo (NVIDIA GPU Accelerated)
echo ========================================================
echo Web UI will be accessible at: http://localhost:7860
echo.
wsl -d Ubuntu bash -c "cd '/mnt/c/College/SEM 7/Modern 3D Vision Techniques/Dust3r/dust3r' && ~/.venvs/dust3r/bin/python demo.py --weights '../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth' --image_size 224 --device cuda --server_name 0.0.0.0 --server_port 7860"
pause
