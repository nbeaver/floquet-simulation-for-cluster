set target_dir=%userprofile%\local\anaconda3\envs\data-analysis\Lib\site-packages
set source_dir=%CD%
set filename=esdr_floquet_lib.py
mklink %target_dir%\%filename% %source_dir%\%filename%
pause