import requests
from bs4 import BeautifulSoup

url = 'http://172.17.1.2/aurocrmis/(S(pneupfk43cougcyyhm1fq1qh))/projectongoinglist.aspx'

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    'Accept-Language':'en-GB,en;q=0.5',
    'Referer':'https://google.com',
    'DNT':'1',
    }

response = requests.get(url,headers = headers)
final_data = []
def getRequiredData(url):
    response = requests.get(url=url)

    soup = BeautifulSoup(response.text, 'html.parser')

    divs = soup.find_all('div', class_='row card-body')

    for div in divs:
        inner_divs = div.find_all('div', class_=['col-lg-4 col-md-4 col-sm-4'])
        
        for inner_div in inner_divs:
            p_tags = inner_div.find_all('p')

            if len(p_tags) > 1:
                p_address = p_tags[0]
                # print((p_address))

                addresses = p_address.find('span').get_text()
                print(addresses)
                print(type(addresses))
                lat, lon = get_geotags(addresses)
