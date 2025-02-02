# TBMCS
The implementation of paper :

Real-time Tear Film Break-up Measurement Based on Multi-task Collaborative System for Dry Eye Instrument



## Installation

1. prepare a python environment:

   ```python
   conda create -n lmv python=3.8 -y
   pip install -r requirements.txt
   ```

2. install pytorch

   ```python
   # for Windows and Linux
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
   
   # for macOS
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
   ```

3. install mmcv

   ```python
   pip install -U openmim
   mim install mmcv
   ```

   
