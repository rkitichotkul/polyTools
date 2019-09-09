# polyTools
Tools for analysis of polymer simulation

polyTools module contains functions for analysis of polymer in Spakowitz Lab's simulation code. This module is still under development. Simply import polyTools to use the tools. 
```python=
import sys
sys.path.append('directory/')
import polyTools
```
Description of all functions can be found in the module. Some analysis functions include ...
- end-to-end distribution
- tangent correlator
- radius of gyration
- area of projection of a ring polymer on xy plane
- area of projection of the convex hull of a ring polymer on xy plane

This module does not support results from parallel tempering yet (the file suffix 'v0' might need to be modified).

All useful functions are in polyTools.py. Other files contain stuff I use for my own work.