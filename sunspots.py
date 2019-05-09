import urllib2

data = urllib2.urlopen("https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/sunspot.long.data").read()
data = data.split("\n") # then split it into lines

for line in data:
    print(line)