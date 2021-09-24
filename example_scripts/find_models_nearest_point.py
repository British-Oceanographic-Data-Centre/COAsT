import coast as cst
import os.path as path

dn_files = "N:\\COAsT\\example_files\\"

anotherpath = "l:\\users\\sgl\\immerse\\data\\"
# The above is a temporary location. Adjust this to access file below
filename = "so_Omon_IPSL-CM5A-LR_historical_r1i1p1_200001-200512.nc"

myconfigfile = path.join(dn_files, "griddedOmonDB.json")
filedata = path.join(anotherpath, filename)
sci = cst.Gridded(filedata, config=myconfigfile)

[j1, i1] = sci.find_j_i(50, -9)
print("j1 = " + str(j1))
print("i1 = " + str(i1))
[j1, i1] = sci.find_j_i(-310, -9)  # Same point on globe gives same result
print("j1 = " + str(j1))
print("i1 = " + str(i1))
