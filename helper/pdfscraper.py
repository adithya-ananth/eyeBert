#!pip install pdfminer.six

import requests
from pdfminer.high_level import extract_pages, extract_text

DOWNLOAD_PATH = "C:/Main/Development/Ongoing/Project_Eye/datadump/file.pdf"
URL = "http://172.17.1.2/sikshagauge/resources/Achieving_Excellence_in_Cataract_Surgery_Chapter_11.pdf"

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    'Accept-Language':'en-GB,en;q=0.5',
    'Referer':'https://google.com',
    'DNT':'1',
    }

def fetch_file(url,download_path,headers):
  try:
    response = requests.get(url,headers=headers)
    if response.status_code==200:
      with open(download_path,'wb') as file:
        file.write(response.content)
      print("File successfully retrieved")
    else:
      print(f"Failed. Status Code:{response.status_code}")
  except Exception as e:
    print(f"Error occured : {e}")

fetch_file(URL,DOWNLOAD_PATH,headers)

raw_data = extract_text(DOWNLOAD_PATH)

with open("C:/Main/Development/Ongoing/Project_Eye/datadump/file.txt","w") as f:
  f.write(raw_data)
f.close()
