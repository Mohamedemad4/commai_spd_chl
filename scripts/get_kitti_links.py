import requests as req
#from bs4 import BeautifulSoup
import pprint
import os
'''
b_url="http://www.cvlibs.net/datasets/kitti/raw_data.php?type={0}"

cats=["city","residential","campus","road"]

all_data_files=[]
all_calib_files=[]
for i in cats:
    c=req.get(b_url.format(i)).content
    links=[]
    soup = BeautifulSoup(c, "lxml")
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    calib_files=[i for i in links if i.endswith("calib.zip")]
    data_files=[i for i in links if i.endswith("sync.zip")]
    all_data_files+=data_files
    all_calib_files+=calib_files


all_calib_files=list(set(all_calib_files)) #rm all duplicates
all_data_files=list(set(all_data_files))

pprint.pprint(all_calib_files)
pprint.pprint(all_data_files)
'''
#os.system("wget -O data.zip -c https://t.co/rUsZvwyntj?amp=1 ")
#os.system("tar xvf data.zip")
for i in os.listdir("."):
    if i.endswith(".zip"):
        print(i)
        os.system("/usr/bin/unzip -q  {0}".format(i))
        os.remove(i)
#for i in all_data_files+all_calib_files:
 #   os.system("wget -c {0} -O {1}".format(i,i.split("/")[-1]))
 #   print(i.split("/")[-1])
 #   os.system("unzip -q --delete {0}".format(i.split("/"))[-1])
    #os.remove(i.split("/")[-1])
