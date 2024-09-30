#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
import requests
import xml.etree.ElementTree as ET




def fetchXML(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup

def getXML(url):
    soup = fetchXML(url)
    return soup.prettify()

def getXMLText(url):
    soup = fetchXML(url)
    return soup.get_text()

def getXMLTitle(url):
    soup = fetchXML(url)
    return soup.title.get_text()

def getXMLLinks(url):
    soup = fetchXML(url)
    links = []
    for link in soup.find_all("a"):
        links.append(link.get("href"))
    return links

def getXMLLinksText(url):
    soup = fetchXML(url)
    links = []
    for link in soup.find_all("a"):
        links.append(link.get_text())
    return links

def getXMLLinksTitle(url):
    soup = fetchXML(url)
    links = []
    for link in soup.find_all("a"):
        links.append(link.get("title"))
    return links

def get_all_XML_urls(base_url):
    # get collections
    collections = getXMLLinks(base_url)

    # get full urls for links
    collections_urls = [base_url + link for link in collections]

    # get first and second collections into a list for sampling/testing
    collections_urls = collections_urls

    # loop through each collection to get their corresponding years; not each collection has the same years available
    urls_master = []
    for collection_url in collections_urls:    
        
        # get the years for each collection and their urls
        years = getXMLLinks(collection_url)
        collectionYear_urls = [collection_url + year for year in years]

        # loop through each year to get the xml links
        for collectionYear_url in collectionYear_urls:

            # get the final xml links for each year for the end of urls
            xmlItems = getXMLLinks(collectionYear_url)
            full_urls = [collectionYear_url + item for item in xmlItems]
            
            # add the full urls to the master list
            urls_master.append(full_urls)

            # add random sleep to avoid getting blocked/detect; between 1-5 seconds
            time.sleep(random.randint(7, 10))
        
        # add random sleep to avoid getting blocked/detect; between 1-5 seconds
        time.sleep(random.randint(7, 10))

    # flatten master list so it is not a nested list
    urls_master = [item for sublist in urls_master for item in sublist]
        
    return urls_master

def get_collection(url):
    collection = url.split("/")[-3]
    return collection

def get_url_xml(url):
     # rate limit(s) are 10 requests per minute

    # raw xml
    raw_xml = getXML(url)
    time.sleep(random.randint(7, 10))

    # raw uncleaned text inside the xml
    raw_text = getXMLText(url)
    time.sleep(random.randint(7, 10))

    return raw_xml, raw_text

def get_lccn(url):

    # strip url of everything but numbers
    lccn = ''.join(filter(str.isdigit, url))

    return lccn


def get_catalog_record_url(url):

    # Send a GET request to fetch the XML content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the XML content
        root = ET.fromstring(response.content)

        # Define the namespace
        namespace = {'ead': 'http://ead3.archivists.org/schema/'}

        # Find the 'ref' element inside 'controlnote' with id 'lccnNote'
        ref_element = root.find('.//ead:controlnote[@id="lccnNote"]/ead:p/ead:ref', namespace)

        # Extract and print the URL
        if ref_element is not None:
            catalog_record_url = ref_element.get('href')

            # add base '/marcxml' to the catalog record URL
            catalog_record_url = catalog_record_url + "/marcxml"

            return catalog_record_url

        else:
            catalog_record_url = f"Failed to fetch XML content. Status code: {response.status_code}"

    else:
        catalog_record_url = f"Failed to fetch XML content. Status code: {response.status_code}"

    # random sleep to avoid getting blocked/detected
    time.sleep(random.randint(1, 3))

    return catalog_record_url

def get_digital_url(row):
    baseurl = row.get("digital_content_baseurl")
    baseurl2 = row.get("digital_content_baseurl2")
    
    if pd.notna(row.get("marc_xml")) and pd.notna(baseurl):
        match = pd.Series(row["marc_xml"]).str.extract(rf'({baseurl}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]
    
    if pd.notna(row.get("ead_xml")) and pd.notna(baseurl):
        match = pd.Series(row["ead_xml"]).str.extract(rf'({baseurl}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]
    
    if pd.notna(row.get("marc_xml")) and pd.notna(baseurl2):
        match = pd.Series(row["marc_xml"]).str.extract(rf'({baseurl2}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]

    if pd.notna(row.get("ead_xml")) and pd.notna(baseurl2):
        match = pd.Series(row["ead_xml"]).str.extract(rf'({baseurl2}[^ ]*)')
        if not match.isna().values.any():
            return match[0].values[0]

    return "No url found"





# get start time
start_time = datetime.now()
print(start_time)

base_url = "https://findingaids.loc.gov/exist_collections/ead3master/"
urls_master = get_all_XML_urls(base_url)

# save to csv
df = pd.DataFrame(urls_master, columns=["urls"])

df = df.head(5)

# get the catalog record urls for each url in the urls_master.csv
df["catalog_marcRecord_url"] = df["urls"].apply(get_catalog_record_url)

# get the raw xml and text for the ead xml
df["ead_xml"], df["ead_xmlText"] = zip(*df["urls"].map(get_url_xml))

# get the raw xml and text for the marc catalog record xml
df["marc_xml"], df["marc_xmlText"] = zip(*df["catalog_marcRecord_url"].map(get_url_xml))

# get the lccn from the catalog record url
df["lccn"] = df["catalog_marcRecord_url"].apply(get_lccn)

# get collection for each url in urls
df["collection"] = df["urls"].apply(get_collection)

# Construct digital content base url that can be found in the ead_xml or marc_xml
df["digital_content_baseurl"] = df["collection"].apply(lambda x: f"http://hdl.loc.gov/loc.{x}/coll{x}")
df["digital_content_baseurl2"] = df["collection"].apply(lambda x: f"https://hdl.loc.gov/loc.{x}/coll{x}")

# column to check if digital content base url is present in the ead_xml or marc_xml (True/False)
df["digital_content"] = df.apply(
    lambda row: (row["digital_content_baseurl"] in row["ead_xml"]) or 
                (row["digital_content_baseurl"] in row["marc_xml"]) or
                (row["digital_content_baseurl2"] in row["ead_xml"]) or
                (row["digital_content_baseurl2"] in row["marc_xml"]),
    axis=1
)

# get full digital content url
df["digital_content_fullurl"] = df.apply(get_digital_url, axis=1)

# get subset of all rows where digital_content is True
df_digitals = df[df["digital_content"] == True] # 158

# save to csv
df.to_csv("xml_master.csv", index=False)
df_digitals.to_csv("xml_digital_urls.csv", index=False)

print(len(df))
print(len(df_digitals))
print(df_digitals.head())

# get end time
end_time = datetime.now()

# get total time
total_time = end_time - start_time
print(f"Total time: {total_time}")













#%%
import pandas as pd

# read in the csv
df = pd.read_csv("xml_master3.csv")
# df.head()

# get first ead_xmlText
#print(df["ead_xmlText"][0])

# get first marc_xmlText
#print(df["marc_xmlText"][0])

# get subset where digital_content is True
df_digitals = df[df["digital_content"] == True]
df_digitals.head()









