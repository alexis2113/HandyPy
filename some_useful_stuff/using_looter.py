
#a crawler for crawling bitcoins information... some old stuff

import urllib
from urllib import request
import re
#pretend as firefox and acquire html data
headers ={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
req = urllib.request.Request(url='https://www.feixiaohao.com/list_2.html', headers=headers)
html_data=urllib.request.urlopen(req).read().decode('utf-8')
pattern = re.compile('alt=+([A-Z]*?)>',re.M)
coinlist = re.findall(pattern,html_data)
coinlist=list(set(coinlist))
#delete replicate

import re
import looter as lt

from lxml import etree
from pprint import pprint
from concurrent import futures

#Really don't know why I use looter.....

proofpattern = re.compile('ProofType":"+(.*?)"',re.M)
algopattern= re.compile('Algorithm":"+(.*?)"',re.M)
maxpattern=re.compile('"TotalCoinSupply":"+(.*?)"',re.M)

#make a container

coininfo={}
coininfo['name']=[]
coininfo['proof']=[]
coininfo['algo']=[]
coininfo['total']=[]

#make a crawler
def mycrawl(url):
    try:
        res = lt.send_request(url)
        res.encoding = 'utf-8'
        webcode=res.text
        coinproof= re.findall(proofpattern,webcode)
        coinalgorithm = re.findall(algopattern,webcode)
        coinmax= re.findall(maxpattern,webcode)
        coininfo['name'].append(url)
        coininfo['algo'].append(coinalgorithm)
        coininfo['proof'].append(coinproof)
        coininfo['total'].append(coinmax)
    except:
        logging.exception('error')
        coininfo['name'].append(url)
        coininfo['algo'].append('error')
        coininfo['proof'].append('error')
        coininfo['total'].append('error')

#some multiprocessing stuff... really useful!

C1=coinlist[50:]

if __name__ == '__main__':
    tasklist = [f'https://www.cryptocompare.com/coins/{n}/overview/USD' for n in C1]
    with futures.ThreadPoolExecutor(40) as executor:
        executor.map(mycrawl,tasklist)


