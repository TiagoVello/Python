from biosig import *

HDR = sopen('gdf2test.gdf')
A = sread(HDR, -1, 1)
B = sread(HDR, 0, 10)
HDR = sclose(HDR)

